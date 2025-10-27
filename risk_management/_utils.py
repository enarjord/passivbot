"""Shared helper utilities for risk management modules."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Tuple

__all__ = [
    "json_default",
    "stringify_payload",
    "first_float",
    "coerce_float",
    "coerce_int",
    "normalize_position_side",
    "extract_position_details",
]


def json_default(value: Any) -> Any:
    """Coerce non-serialisable objects into JSON-compatible types."""

    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def stringify_payload(payload: Any) -> str:
    """Return a JSON string representation for logging purposes."""

    try:
        return json.dumps(payload, ensure_ascii=False, default=json_default, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return repr(payload)


def first_float(*values: Any) -> Optional[float]:
    """Return the first value that can be coerced into ``float``."""

    for value in values:
        candidate = value
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[0]
        try:
            return float(candidate)
        except (TypeError, ValueError):
            if isinstance(candidate, str):
                try:
                    return float(candidate.strip())
                except (TypeError, ValueError):
                    continue
            continue
    return None


def coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` converted to ``float`` when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            try:
                return float(value.strip())
            except (TypeError, ValueError):
                return None
    return None


def coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` converted to ``int`` when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except (TypeError, ValueError):
                return None
    return None


def normalize_position_side(value: Any) -> Optional[str]:
    """Return the normalised hedge-mode side when available."""

    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in {"LONG", "SHORT", "BOTH"}:
            return candidate
    return None


def extract_position_details(position: Mapping[str, Any]) -> Tuple[Optional[str], Optional[int], bool]:
    """Return the detected hedge side, index, and whether the side was explicit."""

    position_side = normalize_position_side(position.get("positionSide"))
    side_explicit = position_side is not None
    if position_side is None:
        alt_side = normalize_position_side(position.get("position_side"))
        if alt_side is not None:
            position_side = alt_side
            side_explicit = True

    def _position_idx_from(obj: Any) -> Optional[int]:
        if not isinstance(obj, Mapping):
            return None
        raw_idx = obj.get("positionIdx", obj.get("position_idx"))
        if raw_idx is None:
            return None
        try:
            value = int(float(raw_idx))
        except (TypeError, ValueError):
            return None
        if value in {0, 1, 2}:
            return value
        return None

    position_idx = _position_idx_from(position)
    info = position.get("info")
    if position_idx is None and isinstance(info, Mapping):
        position_idx = _position_idx_from(info)

    if position_side is None and isinstance(info, Mapping):
        info_side = normalize_position_side(info.get("positionSide") or info.get("position_side"))
        if info_side is not None:
            position_side = info_side
            side_explicit = True

    if position_side is None and position_idx in {1, 2}:
        position_side = "LONG" if position_idx == 1 else "SHORT"

    return position_side, position_idx, side_explicit

