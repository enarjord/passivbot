from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths


@dataclass(frozen=True)
class OptimizationShape:
    bounds: Tuple[Bound, ...]
    key_paths: Tuple[tuple[str, tuple[str, ...]], ...]
    sig_digits: int | None


def build_optimization_shape(config: dict) -> OptimizationShape:
    return OptimizationShape(
        bounds=tuple(extract_bounds_tuple_list_from_config(config)),
        key_paths=tuple(get_optimization_key_paths(config)),
        sig_digits=config.get("optimize", {}).get("round_to_n_significant_digits", 6),
    )
