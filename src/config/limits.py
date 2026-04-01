import json
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from .metrics import canonicalize_limit_name, canonicalize_metric_name


def parse_limits_string(limits_str: Union[str, dict]) -> dict:
    if not limits_str:
        return {}
    if isinstance(limits_str, dict):
        return limits_str
    tokens = limits_str.replace(":", "").split("--")
    result = {}
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            key, value = token.split()
            result[key] = float(value)
        except ValueError:
            raise ValueError(f"Invalid limits format for token: {token}")
    return result


def _ensure_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _restore_numeric_precision(value: Optional[float]) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, float) and math.isfinite(value):
        rounded = round(value)
        if abs(value - rounded) < 1e-12:
            return int(rounded)
    return value


def _extract_range(payload: Any) -> Optional[Tuple[float, float]]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        low = payload.get("low")
        high = payload.get("high")
        if low is None:
            low = payload.get("min")
        if low is None:
            low = payload.get("start")
        if high is None:
            high = payload.get("max")
        if high is None:
            high = payload.get("end")
        if low is None or high is None:
            return None
        low_f = _ensure_float(low)
        high_f = _ensure_float(high)
        if low_f is None or high_f is None:
            return None
        return (min(low_f, high_f), max(low_f, high_f))
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        low_f = _ensure_float(payload[0])
        high_f = _ensure_float(payload[1])
        if low_f is None or high_f is None:
            return None
        return (min(low_f, high_f), max(low_f, high_f))
    return None


def _normalize_penalize_if(value: Any) -> str:
    if value is None:
        raise ValueError("limits entries must include 'penalize_if'.")
    token = str(value).strip().lower()
    mapping = {
        ">": "greater_than",
        "gt": "greater_than",
        "greater": "greater_than",
        "greater_than": "greater_than",
        "above": "greater_than",
        "<": "less_than",
        "lt": "less_than",
        "lower": "less_than",
        "less": "less_than",
        "less_than": "less_than",
        "below": "less_than",
        "outside": "outside_range",
        "outside_range": "outside_range",
        "out_of_range": "outside_range",
        "inside": "inside_range",
        "inside_range": "inside_range",
        "auto": "auto",
    }
    normalized = mapping.get(token)
    if not normalized:
        raise ValueError(f"Unsupported penalize_if value '{value}'.")
    return normalized


def _normalize_limit_entry(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Each limit entry must be a dict, got {type(entry).__name__}.")
    payload = deepcopy(entry)
    metric = payload.get("metric") or payload.get("name")
    if not metric:
        raise ValueError("Limit entries must include a 'metric' field.")
    metric = canonicalize_metric_name(str(metric))
    enabled = payload.get("enabled")
    enabled = True if enabled is None else bool(enabled)
    raw_penalize_if = payload.get("penalize_if")
    if raw_penalize_if is None and not enabled:
        penalize_if = "greater_than"
    else:
        penalize_if = _normalize_penalize_if(raw_penalize_if)
    stat = payload.get("stat") or payload.get("field")
    normalized_stat: Optional[str] = None
    if stat is not None:
        stat = str(stat).lower()
        if stat not in {"min", "max", "mean", "std"}:
            raise ValueError(f"Unsupported stat '{stat}' for limit on {metric}.")
        normalized_stat = stat
    result: Dict[str, Any] = {"metric": metric, "penalize_if": penalize_if}
    if normalized_stat:
        result["stat"] = normalized_stat
    if penalize_if in {"greater_than", "less_than", "auto"}:
        bound = payload.get("value")
        if bound is None:
            bound = payload.get("threshold")
        if bound is None:
            bound = payload.get("bound")
        if bound is None and not enabled:
            bound = 0.0
        numeric_bound = _ensure_float(bound)
        if numeric_bound is None:
            raise ValueError(f"Limit for {metric} requires a numeric 'value'.")
        result["value"] = _restore_numeric_precision(numeric_bound)
    elif penalize_if in {"outside_range", "inside_range"}:
        range_payload = payload.get("range")
        if range_payload is None:
            range_payload = payload.get("values")
        if range_payload is None:
            range_payload = payload.get("bounds")
        if range_payload is None and isinstance(payload.get("value"), (list, tuple)):
            range_payload = payload.get("value")
        if range_payload is None and not enabled:
            range_payload = [0.0, 0.0]
        bounds = _extract_range(range_payload)
        if bounds is None:
            raise ValueError(f"Limit for {metric} requires a two-value 'range'.")
        result["range"] = [
            _restore_numeric_precision(bounds[0]),
            _restore_numeric_precision(bounds[1]),
        ]
    else:
        raise ValueError(f"Unsupported penalize_if '{penalize_if}' for {metric}.")
    return result


def _numeric_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def _range_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    if len(a) != len(b):
        return False
    return all(_numeric_equal(x, y, tol=tol) for x, y in zip(a, b))


def _normalize_limit_entry_preserve_extras(entry: Any) -> Dict[str, Any]:
    normalized = _normalize_limit_entry(entry)
    if not isinstance(entry, dict):
        return normalized
    for key, value in entry.items():
        if key not in normalized:
            normalized[key] = deepcopy(value)
    return normalized


def _is_canonical_limit_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        normalized = _normalize_limit_entry(entry)
    except Exception:
        return False
    for key, norm_val in normalized.items():
        raw_val = entry.get(key)
        if key == "range":
            if not _range_equal(raw_val, norm_val):
                return False
        else:
            if not _numeric_equal(raw_val, norm_val):
                return False
    return True


def _legacy_limits_dict_to_entries(limits_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key, value in limits_dict.items():
        canonical_key = canonicalize_limit_name(key)
        if canonical_key.startswith("penalize_if_greater_than_"):
            metric = canonical_key[len("penalize_if_greater_than_") :]
            penalty = "greater_than"
        elif canonical_key.startswith("penalize_if_lower_than_"):
            metric = canonical_key[len("penalize_if_lower_than_") :]
            penalty = "less_than"
        else:
            metric = canonical_key
            penalty = "auto"
        numeric_value = _ensure_float(value)
        if numeric_value is None:
            raise ValueError(f"Limit '{key}' must have a numeric value.")
        entries.append({"metric": metric, "penalize_if": penalty, "value": numeric_value})
    return entries


def normalize_limit_entries(
    raw_limits: Union[str, List[dict], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if raw_limits is None:
        return []
    parsed = raw_limits
    if isinstance(raw_limits, str):
        stripped = raw_limits.strip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = parse_limits_string(stripped)
        else:
            parsed = parse_limits_string(stripped)
    elif isinstance(raw_limits, dict):
        parsed = deepcopy(raw_limits)
    elif isinstance(raw_limits, list):
        parsed = deepcopy(raw_limits)
    else:
        raise ValueError(f"Unsupported limits format: {type(raw_limits).__name__}")

    if isinstance(parsed, dict):
        entries: List[Dict[str, Any]] = _legacy_limits_dict_to_entries(parsed)
    elif isinstance(parsed, list):
        entries = parsed
    else:
        raise ValueError(f"Unsupported parsed limits payload: {type(parsed).__name__}")

    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        normalized.append(_normalize_limit_entry_preserve_extras(entry))
    return normalized


def _resolve_optimize_limits_for_load(
    *,
    raw_optimize_limits: Any,
    raw_optimize_limits_present: bool,
    template_limits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    if not raw_optimize_limits_present:
        return deepcopy(template_limits), "template_default"

    if isinstance(raw_optimize_limits, list):
        if all(_is_canonical_limit_entry(entry) for entry in raw_optimize_limits):
            return deepcopy(raw_optimize_limits), "preserved_canonical"
        try:
            return normalize_limit_entries(raw_optimize_limits), "normalized_legacy"
        except Exception:
            return deepcopy(template_limits), "fallback_template"

    if isinstance(raw_optimize_limits, (str, dict)):
        try:
            return normalize_limit_entries(raw_optimize_limits), "normalized_legacy"
        except Exception:
            return deepcopy(template_limits), "fallback_template"

    return deepcopy(template_limits), "fallback_template"
