"""Retire implicit GPU history windows without silently changing search intent."""

import logging

from passivbot_exceptions import GPUScreeningMigrationError


def migrate_gpu_screening(config: dict, *, tracker=None) -> None:
    gpu = config.get("optimize", {}).get("gpu")
    if not isinstance(gpu, dict) or "successive_halving" not in gpu:
        return
    legacy = gpu["successive_halving"]
    if legacy is not None and not isinstance(legacy, dict):
        raise GPUScreeningMigrationError(
            "optimize.gpu.successive_halving has been removed; remove this field "
            "and configure optimize.gpu.screening.scenarios instead. See docs/optimizing.md."
        )
    enabled = (legacy or {}).get("enabled", False)
    # Accept the false spellings used by historical CLI/config inputs, but never
    # interpret an unknown value as permission to discard an active policy.
    disabled = enabled is False or enabled == 0 or (
        isinstance(enabled, str) and enabled.strip().lower() in {"false", "n", "no", "0"}
    )
    if not disabled:
        raise GPUScreeningMigrationError(
            "optimize.gpu.successive_halving is no longer supported when enabled. "
            "Automatic migration would change the screening windows. "
            "Define explicit date ranges in backtest.scenarios, enable backtest.suite_enabled, "
            "and select their labels in optimize.gpu.screening.scenarios; copy "
            "survival_fraction and min_survivors into optimize.gpu.screening, then remove "
            "successive_halving. Use screening.scenarios=[] to disable screening. "
            "Start a new optimization; active halving checkpoints cannot resume with this policy. "
            "See docs/optimizing.md (GPU scenario screening)."
        )
    del gpu["successive_halving"]
    if tracker is not None:
        tracker.remove(["optimize", "gpu", "successive_halving"], legacy)
    logging.warning(
        "Removed disabled legacy optimize.gpu.successive_halving. "
        "Save the normalized config to remove this warning."
    )
