"""Normalize scenario override documents without importing simulation runtimes."""

from copy import deepcopy
from typing import Any, Dict, Optional

_ATOMIC_SCENARIO_OVERRIDE_ROOTS = frozenset({"coin_overrides"})


def normalize_scenario_overrides(
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Flatten nested override documents while preserving atomic dynamic mappings."""
    normalized: Dict[str, Any] = {}

    def visit(mapping: Dict[str, Any], prefix: tuple[str, ...] = ()) -> None:
        for raw_key, value in mapping.items():
            if not isinstance(raw_key, str):
                raise ValueError("Scenario override keys must be strings")
            key = raw_key.strip()
            if not key:
                raise ValueError("Scenario override keys must not be empty")
            path = (*prefix, key)
            root = path[0].split(".", 1)[0]
            if (
                isinstance(value, dict)
                and "." not in key
                and root not in _ATOMIC_SCENARIO_OVERRIDE_ROOTS
            ):
                visit(value, path)
                continue
            dotted_path = ".".join(path)
            if dotted_path in normalized:
                raise ValueError(
                    f"Scenario override path {dotted_path!r} is defined more than once"
                )
            normalized[dotted_path] = deepcopy(value)

    if overrides:
        visit(overrides)
    return normalized
