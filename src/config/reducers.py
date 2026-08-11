from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


SUPPORTED_REDUCERS = frozenset({"min", "max", "mean", "std", "median"})
REDUCER_ALIASES = ("reducer", "aggregate", "stat", "scenario_stat")
LIMIT_REDUCER_ALIASES = (*REDUCER_ALIASES, "field")


def normalize_reducer(value: Any, *, path: str) -> str | None:
    if value is None:
        return None
    reducer = str(value).strip().lower()
    if reducer not in SUPPORTED_REDUCERS:
        raise ValueError(
            f"{path} must be one of {sorted(SUPPORTED_REDUCERS)}, got {value!r}"
        )
    return reducer


def reducer_from_aliases(
    payload: Mapping[str, Any],
    *,
    path: str,
    aliases: Sequence[str] = REDUCER_ALIASES,
) -> tuple[str | None, bool]:
    present = [(key, payload[key]) for key in aliases if key in payload]
    if not present:
        return None, False

    normalized = [
        (key, normalize_reducer(value, path=f"{path}.{key}")) for key, value in present
    ]
    distinct = {value for _key, value in normalized}
    if len(distinct) > 1:
        rendered = ", ".join(f"{key}={payload[key]!r}" for key, _value in normalized)
        raise ValueError(f"{path} has conflicting reducer aliases: {rendered}")
    return normalized[0][1], True


def reducer_mapping_from_aliases(
    payload: Mapping[str, Any],
    *,
    path: str,
    aliases: Sequence[str] = REDUCER_ALIASES,
) -> tuple[dict[str, Any] | None, bool]:
    present = [(key, payload[key]) for key in aliases if key in payload]
    if not present:
        return None, False

    normalized: list[tuple[str, dict[str, Any]]] = []
    for key, raw_mapping in present:
        if not isinstance(raw_mapping, Mapping):
            raise ValueError(f"{path}.{key} must be a mapping")
        mapping = {}
        for metric, value in raw_mapping.items():
            metric_path = f"{path}.{key}.{metric}"
            reducer = normalize_reducer(value, path=metric_path)
            if reducer is None:
                raise ValueError(f"{metric_path} must be a reducer name, not null")
            mapping[str(metric)] = reducer
        normalized.append((key, mapping))

    first = normalized[0][1]
    if any(mapping != first for _key, mapping in normalized[1:]):
        rendered = ", ".join(f"{key}={payload[key]!r}" for key, _mapping in normalized)
        raise ValueError(f"{path} has conflicting reducer aliases: {rendered}")
    return deepcopy(first), True


def canonicalize_reducer_mapping(
    payload: dict[str, Any],
    *,
    path: str,
    aliases: Sequence[str] = REDUCER_ALIASES,
) -> bool:
    mapping, present = reducer_mapping_from_aliases(payload, path=path, aliases=aliases)
    if not present:
        return False
    for alias in aliases:
        payload.pop(alias, None)
    payload["reducer"] = mapping
    return True
