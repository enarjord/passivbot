from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config.scoring import ObjectiveSpec, dominates_objectives, extract_objective_specs, from_engine_value


@dataclass(frozen=True)
class ParetoPoint:
    hash_id: str
    objectives: Tuple[float, ...]
    violation: float = 0.0


def extract_objectives(
    entry: Dict[str, Any], scoring_keys: Optional[Sequence[str]] = None
) -> Tuple[Tuple[float, ...], List[str]]:
    """
    Extract raw objective values from a result entry.
    Ordered by scoring keys/specs if provided, otherwise by sorted named keys.
    Legacy w_i engine-space payloads are converted back to raw values when scoring
    metadata is available.
    """
    metrics_block = entry.get("metrics") or {}
    objectives_map = metrics_block.get("objectives", metrics_block) or {}
    specs = extract_objective_specs(scoring_keys or entry.get("optimize", {}).get("scoring", []))
    if specs:
        values: list[float] = []
        keys: list[str] = []
        for idx, spec in enumerate(specs):
            if spec.metric in objectives_map:
                value = objectives_map.get(spec.metric)
            else:
                value = objectives_map.get(f"w_{idx}")
                if value is not None:
                    value = from_engine_value(spec, float(value))
            values.append(value)
            keys.append(spec.metric)
        return tuple(values), keys

    keys = sorted(k for k in objectives_map if not str(k).startswith("w_"))
    if not keys:
        keys = sorted(k for k in objectives_map if str(k).startswith("w_"))
    objectives = tuple(objectives_map.get(key) for key in keys)
    return objectives, keys


def extract_violation(entry: Dict[str, Any]) -> float:
    metrics_block = entry.get("metrics") or {}
    try:
        return float(metrics_block.get("constraint_violation") or 0.0)
    except Exception:
        return 0.0


def dominates_with_violation(
    obj_a: Sequence[float],
    viol_a: float,
    obj_b: Sequence[float],
    viol_b: float,
    objective_specs: Optional[Sequence[ObjectiveSpec]] = None,
    tol: float = 1e-12,
) -> bool:
    """
    Constraint-aware dominance: lower violation wins ties; otherwise standard Pareto
    dominance using objective directions when provided.
    """
    if np.isclose(viol_a, viol_b, atol=tol, rtol=0.0):
        if objective_specs:
            return dominates_objectives(obj_a, obj_b, objective_specs)
        better_in_one = False
        for a, b in zip(obj_a, obj_b):
            if a < b:
                better_in_one = True
            elif a > b:
                return False
        return better_in_one
    return viol_a < viol_b


def crowding_distances(values: np.ndarray) -> np.ndarray:
    """
    Compute crowding distances for an array of objective vectors (lower is more crowded).
    """
    if values.ndim != 2:
        return np.zeros(len(values))
    n, m = values.shape
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)
    distances = np.zeros(n)
    for col in range(m):
        order = np.argsort(values[:, col])
        distances[order[0]] = distances[order[-1]] = np.inf
        column = values[order, col]
        min_v = column[0]
        max_v = column[-1]
        denom = max_v - min_v
        if denom == 0:
            continue
        normalized = (column[2:] - column[:-2]) / denom
        distances[order[1:-1]] += normalized
    return distances


def prune_front_with_extremes(
    front_hashes: Sequence[str],
    objectives_map: Dict[str, Tuple[float, ...]],
    violations_map: Dict[str, float],
    max_size: int,
) -> List[str]:
    """
    Determine which members to remove to satisfy max_size while always
    retaining extremes (min/max) per objective axis.
    Returns the list of hash_ids to drop.
    """
    if max_size <= 0 or len(front_hashes) <= max_size:
        return []
    objs = [objectives_map[idx] for idx in front_hashes]
    arr = np.asarray(objs, dtype=float)
    required: set[str] = set()
    for dim in range(arr.shape[1]):
        min_idx = int(np.argmin(arr[:, dim]))
        max_idx = int(np.argmax(arr[:, dim]))
        required.add(front_hashes[min_idx])
        required.add(front_hashes[max_idx])

    crowding = crowding_distances(arr)
    scored = list(zip(front_hashes, crowding))
    scored.sort(key=lambda item: item[1])  # lowest crowding removed first

    to_remove: List[str] = []
    for hash_id, _cd in scored:
        if hash_id in required:
            continue
        to_remove.append(hash_id)
        if len(to_remove) >= len(front_hashes) - max_size:
            break
    return to_remove


def compute_ideal(
    values_matrix: np.ndarray,
    mode: str = "min",
    weights=None,
    eps: float = 1e-3,
    pct: float = 10,
    objective_specs: Optional[Sequence[ObjectiveSpec]] = None,
):
    def _require_specs() -> Sequence[ObjectiveSpec]:
        if not objective_specs:
            raise ValueError("objective_specs required for goal-aware ideal computation")
        if len(objective_specs) != values_matrix.shape[1]:
            raise ValueError(
                "objective_specs length must match objective column count "
                f"({len(objective_specs)} != {values_matrix.shape[1]})"
            )
        return objective_specs

    if objective_specs:
        specs = _require_specs()
        mins = values_matrix.min(axis=0)
        maxs = values_matrix.max(axis=0)
        if mode in ["m", "min"]:
            return np.array(
                [
                    maxs[i] if specs[i].goal == "max" else mins[i]
                    for i in range(values_matrix.shape[1])
                ]
            )
        if mode in ["w", "weighted"]:
            if weights is None:
                raise ValueError("weights required")
            ideal = np.array(
                [
                    maxs[i] if specs[i].goal == "max" else mins[i]
                    for i in range(values_matrix.shape[1])
                ]
            )
            anti_ideal = np.array(
                [
                    mins[i] if specs[i].goal == "max" else maxs[i]
                    for i in range(values_matrix.shape[1])
                ]
            )
            return ideal + weights * (anti_ideal - ideal)
        if mode in ["u", "utopian"]:
            ranges = maxs - mins
            return np.array(
                [
                    maxs[i] + eps * ranges[i] if specs[i].goal == "max" else mins[i] - eps * ranges[i]
                    for i in range(values_matrix.shape[1])
                ]
            )
        if mode in ["p", "percentile"]:
            return np.array(
                [
                    np.percentile(values_matrix[:, i], 100.0 - pct)
                    if specs[i].goal == "max"
                    else np.percentile(values_matrix[:, i], pct)
                    for i in range(values_matrix.shape[1])
                ]
            )
        if mode in ["mi", "midrange"]:
            return 0.5 * (mins + maxs)

    # values_matrix:  shape (n_points, n_obj)
    if mode in ["m", "min"]:
        return values_matrix.min(axis=0)

    if mode in ["w", "weighted"]:
        if weights is None:
            raise ValueError("weights required")
        vmin = values_matrix.min(axis=0)
        vmax = values_matrix.max(axis=0)
        return vmin + weights * (vmax - vmin)

    if mode in ["u", "utopian"]:
        mins = values_matrix.min(axis=0)
        ranges = values_matrix.ptp(axis=0)
        return mins - eps * ranges  # ε-shift

    if mode in ["p", "percentile"]:
        return np.percentile(values_matrix, pct, axis=0)

    if mode in ["mi", "midrange"]:
        return 0.5 * (values_matrix.min(axis=0) + values_matrix.max(axis=0))

    if mode in ["g", "geomedian"]:
        z = values_matrix.mean(axis=0)
        for _ in range(10):
            d = np.linalg.norm(values_matrix - z, axis=1)
            w = np.where(d > 0, 1.0 / d, 0.0)
            z_new = (values_matrix * w[:, None]).sum(axis=0) / w.sum()
            if np.allclose(z, z_new, atol=1e-9):
                break
            z = z_new
        return z

    raise ValueError(f"unknown mode {mode}")
