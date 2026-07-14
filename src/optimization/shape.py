from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths
from optimization.fine_tune_anchors import ANCHOR_GENE_KEY, get_anchor_plan


@dataclass(frozen=True)
class OptimizationShape:
    bounds: Tuple[Bound, ...]
    key_paths: Tuple[tuple[str, tuple[str, ...]], ...]
    sig_digits: int | None


def build_optimization_shape(config: dict) -> OptimizationShape:
    anchor_plan = get_anchor_plan(config)
    if anchor_plan is not None:
        tunable_keys = set(anchor_plan.get("tunable_keys") or [])
        base_key_paths = tuple(get_optimization_key_paths(config))
        base_bounds = tuple(extract_bounds_tuple_list_from_config(config))
        selected = [
            (key_path, bound)
            for key_path, bound in zip(base_key_paths, base_bounds)
            if key_path[0] in tunable_keys
        ]
        anchor_count = len(anchor_plan.get("anchors") or [])
        anchor_bound = Bound(0.0, float(max(0, anchor_count - 1)), 1.0)
        return OptimizationShape(
            bounds=(anchor_bound, *(bound for _, bound in selected)),
            key_paths=((ANCHOR_GENE_KEY, (ANCHOR_GENE_KEY,)), *(key_path for key_path, _ in selected)),
            sig_digits=config.get("optimize", {}).get("round_to_n_significant_digits", 6),
        )
    return OptimizationShape(
        bounds=tuple(extract_bounds_tuple_list_from_config(config)),
        key_paths=tuple(get_optimization_key_paths(config)),
        sig_digits=config.get("optimize", {}).get("round_to_n_significant_digits", 6),
    )
