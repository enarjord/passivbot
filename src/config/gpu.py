"""Canonical GPU scenario-screening policy, independent of optional GPU imports."""

import json
import math

GPU_SCREENING_DEFAULTS = {"scenarios": [], "survival_fraction": 0.1, "min_survivors": 64}


def resolve_gpu_screening(value=None) -> dict:
    screening = dict(GPU_SCREENING_DEFAULTS)
    configured_screening = value
    if configured_screening is not None and not isinstance(configured_screening, dict):
        raise TypeError("optimize.gpu.screening must be an object")
    screening.update(configured_screening or {})
    unknown = sorted(set(screening) - set(GPU_SCREENING_DEFAULTS))
    if unknown:
        raise ValueError("unknown optimize.gpu.screening settings: " + ", ".join(unknown))
    labels = screening["scenarios"]
    if (
        not isinstance(labels, list)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError(
            "optimize.gpu.screening.scenarios must be an array of unique non-empty scenario labels"
        )
    survival_fraction = float(screening["survival_fraction"])
    if not math.isfinite(survival_fraction) or not 0.0 < survival_fraction <= 1.0:
        raise ValueError("optimize.gpu.screening.survival_fraction must be in (0, 1]")
    min_survivors = int(screening["min_survivors"])
    if min_survivors <= 0:
        raise ValueError("optimize.gpu.screening.min_survivors must be greater than zero")
    screening["scenarios"] = list(labels)
    screening.update(survival_fraction=survival_fraction, min_survivors=min_survivors)
    return screening


def parse_screening_scenarios(value: str) -> list[str]:
    """CLI accepts comma-separated labels or a JSON array, including [] to disable."""
    raw = value.strip()
    labels = json.loads(raw) if raw.startswith("[") else [label.strip() for label in raw.split(",")]
    return resolve_gpu_screening({"scenarios": labels})["scenarios"]
