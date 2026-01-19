"""Pareto front extraction for Optuna optimizer."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import optuna

from .config import apply_params_to_config
from .models import Objective


def _select_diverse(
    trials: list["optuna.FrozenTrial"],
    objectives: list[Objective],
    max_best_trials: int,
) -> list["optuna.FrozenTrial"]:
    """Select diverse subset using greedy farthest-point sampling.

    Starts with the solution closest to the ideal point (best overall),
    then greedily adds solutions that maximize minimum distance to the
    already-selected set. Works well for any number of objectives.

    Args:
        trials: List of Pareto-optimal trials
        objectives: List of objectives (used for sign correction)
        max_best_trials: Maximum number of trials to return

    Returns:
        Subset of trials balancing quality and diversity
    """
    n_trials = len(trials)
    if n_trials <= max_best_trials:
        return trials

    n_objectives = len(objectives)

    # Build normalized objective values matrix (n_trials x n_objectives)
    # Convert to "higher is better" space for all objectives
    values = np.array([
        [trial.values[i] * objectives[i].sign for i in range(n_objectives)]
        for trial in trials
    ])

    # Normalize by range to give equal weight to each objective
    ranges = values.max(axis=0) - values.min(axis=0)
    ranges[ranges == 0] = 1
    normalized = values / ranges

    # Compute ideal point and distances to it
    ideal = normalized.max(axis=0)
    ideal_distances = np.sqrt(((normalized - ideal) ** 2).sum(axis=1))

    # Start with solution closest to ideal point
    selected = [int(np.argmin(ideal_distances))]
    remaining = set(range(n_trials)) - set(selected)

    # Greedily add farthest points
    while len(selected) < max_best_trials and remaining:
        # For each remaining point, compute min distance to any selected point
        min_dists = np.full(n_trials, np.inf)
        for idx in remaining:
            for sel_idx in selected:
                dist = np.sqrt(((normalized[idx] - normalized[sel_idx]) ** 2).sum())
                min_dists[idx] = min(min_dists[idx], dist)

        # Select the point with maximum min-distance (farthest from selected set)
        best_idx = max(remaining, key=lambda i: min_dists[i])
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [trials[i] for i in selected]


def _sort_by_ideal_distance(
    trials: list["optuna.FrozenTrial"],
    objectives: list[Objective],
) -> list["optuna.FrozenTrial"]:
    """Sort trials by normalized Euclidean distance to ideal point.

    The ideal point is the best value achieved for each objective on the Pareto front.
    Trials closest to the ideal point (best overall) are sorted first.

    Args:
        trials: List of Pareto-optimal trials
        objectives: List of objectives (used for sign correction)

    Returns:
        Trials sorted by ascending distance to ideal point
    """
    if len(trials) <= 1:
        return trials

    n_objectives = len(objectives)

    # Build objective values matrix (n_trials x n_objectives)
    # Convert to "higher is better" space for all objectives
    values = np.array([
        [trial.values[i] * objectives[i].sign for i in range(n_objectives)]
        for trial in trials
    ])

    # Compute ideal point (best value for each objective)
    ideal = values.max(axis=0)

    # Normalize by range to give equal weight to each objective
    ranges = values.max(axis=0) - values.min(axis=0)
    ranges[ranges == 0] = 1  # Avoid division by zero

    # Compute normalized distance to ideal for each trial
    normalized = (ideal - values) / ranges
    distances = np.sqrt((normalized ** 2).sum(axis=1))

    # Sort by ascending distance (closest to ideal first)
    sorted_indices = np.argsort(distances)

    return [trials[i] for i in sorted_indices]


def trial_to_config(trial: "optuna.FrozenTrial", base_config: dict) -> dict:
    """Merge trial parameters into a base config.

    Args:
        trial: Optuna trial with optimized parameters
        base_config: Base configuration to merge into

    Returns:
        New config dict with trial params applied using proper routing.
    """
    return apply_params_to_config(trial.params, base_config)


def extract_pareto(
    study: "optuna.Study",
    study_dir: Path,
    objectives: list[Objective],
    base_config: dict,
    max_best_trials: int = 200,
) -> None:
    """Extract Pareto front configs from study and write to disk.

    Args:
        study: Optuna study containing completed trials
        study_dir: Directory to write Pareto configs
        objectives: List of Objective objects (used in filenames)
        base_config: Base configuration to merge trial params into
        max_best_trials: Maximum configs to export (uses farthest-point sampling)
    """
    pareto_dir = study_dir / "pareto"
    pareto_dir.mkdir(exist_ok=True)

    trials = study.best_trials
    original_count = len(trials)

    if max_best_trials > 0 and original_count > max_best_trials:
        trials = _select_diverse(trials, objectives, max_best_trials)
        logging.info(f"Pruned Pareto front from {original_count} to {len(trials)} using farthest-point sampling")

    # Sort by distance to ideal point (best overall configs first)
    trials = _sort_by_ideal_distance(trials, objectives)

    for rank, trial in enumerate(trials, start=1):
        metrics_parts = []
        for obj, val in zip(objectives, trial.values):
            # Undo the sign transformation to show actual metric values
            actual_val = val * obj.sign
            metrics_parts.append(f"{obj.metric}{actual_val:.6f}")
        metrics_str = "_".join(metrics_parts)
        # Rank prefix ensures alphabetical sort matches quality order
        filename = f"{rank:04d}_t{trial.number:04d}_{metrics_str}.json"
        config = trial_to_config(trial, base_config)
        (pareto_dir / filename).write_text(json.dumps(config, indent=2))

    logging.info(f"Wrote {len(trials)} Pareto configs to {pareto_dir}")
