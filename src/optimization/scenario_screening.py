"""CPU feasibility and Pareto-diversity selection, independent of simulation."""

import numpy as np
import math


def screening_survivor_count(size, policy):
    return min(size, max(int(policy["min_survivors"]), math.ceil(size * float(policy["survival_fraction"]))))


def normalized_farthest_indices(values: np.ndarray, count: int) -> list[int]:
    values = np.asarray(values, dtype=np.float64)
    if count <= 0 or len(values) == 0:
        return []
    if len(values) <= count:
        return list(range(len(values)))
    low = np.nanmin(values, axis=0)
    span = np.nanmax(values, axis=0) - low
    normalized = (values - low) / np.where(span > 1.0e-12, span, 1.0)
    chosen = [int(np.argmin(np.nanmean(normalized, axis=1)))]
    selected = np.zeros(len(values), dtype=bool)
    selected[chosen[0]] = True
    distance = np.linalg.norm(normalized - normalized[chosen[0]], axis=1)
    for _ in range(count - 1):
        available = np.flatnonzero(~selected)
        available_distances = np.where(
            np.isfinite(distance[available]), distance[available], -np.inf
        )
        index = int(available[int(np.argmax(available_distances))])
        chosen.append(index)
        selected[index] = True
        distance = np.minimum(
            distance, np.linalg.norm(normalized - normalized[index], axis=1)
        )
    return chosen


def screening_survivor_indices(
    objectives: np.ndarray,
    violations: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    """Select a deterministic constraint-aware, Pareto-diverse screening subset."""

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    objectives = np.asarray(objectives, dtype=np.float64)
    violations = np.asarray(violations, dtype=np.float64)
    if objectives.ndim != 2 or len(objectives) != len(violations):
        raise ValueError("screening objectives and violations must align")
    count = min(max(0, int(count)), len(objectives))
    if count == 0:
        return np.empty(0, dtype=np.int64)

    feasible = np.flatnonzero(np.isfinite(violations) & (violations <= 0.0))
    feasible_ids = set(map(int, feasible))
    infeasible = np.asarray(
        sorted(
            (
                int(index)
                for index in range(len(objectives))
                if int(index) not in feasible_ids
            ),
            key=lambda index: (
                (
                    float(violations[index])
                    if np.isfinite(violations[index])
                    else float("inf")
                ),
                index,
            ),
        ),
        dtype=np.int64,
    )
    selected: list[int] = []
    if len(feasible):
        for front_local in NonDominatedSorting().do(objectives[feasible]):
            front = feasible[np.asarray(front_local, dtype=np.int64)]
            remaining = count - len(selected)
            if remaining <= 0:
                break
            if len(front) <= remaining:
                selected.extend(map(int, front))
                continue
            diverse = normalized_farthest_indices(objectives[front], remaining)
            selected.extend(int(front[index]) for index in diverse)
            break
    if len(selected) < count:
        selected.extend(map(int, infeasible[: count - len(selected)]))
    return np.asarray(selected, dtype=np.int64)
