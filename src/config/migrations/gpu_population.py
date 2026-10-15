"""Preserve explicit GPU population choices before obsolete fields are pruned."""

import logging
import math


def _is_unset(value) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in {"", "auto", "none", "null"}
    )


def _legacy_population(value) -> int | None:
    if _is_unset(value):
        return None
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError
        if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
            raise ValueError
        population = int(value)
        if population <= 0:
            raise ValueError
    except (ValueError, OverflowError):
        raise ValueError(
            "optimize.gpu.population_size must be a positive integer or auto; "
            "use optimize.population_size for new configs"
        ) from None
    # The former GPU backend accepted positive values but ran at least eight parents.
    return max(8, population)


def migrate_gpu_population(config: dict, *, tracker=None) -> None:
    optimize = config.get("optimize", {})
    backend = str(optimize.get("backend", "pymoo") or "pymoo").strip().lower()
    gpu = optimize.get("gpu")
    if backend != "gpu" or not isinstance(gpu, dict) or "population_size" not in gpu:
        return
    legacy = gpu["population_size"]
    population = _legacy_population(legacy)
    current = optimize.get("population_size")
    if population is not None and _is_unset(current):
        optimize["population_size"] = population
        if tracker is not None:
            tracker.rename(
                ["optimize", "gpu", "population_size"],
                ["optimize", "population_size"],
                population,
            )
        logging.warning(
            "Migrated optimize.gpu.population_size to optimize.population_size=%d "
            "(legacy minimum eight). Save the normalized config to remove this warning.",
            population,
        )
    else:
        if tracker is not None:
            tracker.remove(["optimize", "gpu", "population_size"], legacy)
        logging.warning(
            "Removed legacy optimize.gpu.population_size=%r; "
            "optimize.population_size=%r remains authoritative. "
            "Save the normalized config to remove this warning.",
            legacy,
            current,
        )
    del gpu["population_size"]
