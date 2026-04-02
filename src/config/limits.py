import json
import shlex
import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import hjson

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


def _parse_jsonish(raw: str) -> Any:
    stripped = str(raw).strip()
    if not stripped or stripped[0] not in "[{" or stripped[-1] not in "]}":
        return None
    try:
        return hjson.loads(stripped)
    except Exception:
        try:
            return json.loads(stripped)
        except Exception:
            return None


def _parse_cli_bool(token: str) -> bool:
    normalized = str(token).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean token {token!r}; expected true/false.")


def _parse_cli_range_token(token: str) -> List[Union[int, float]]:
    parsed = _parse_jsonish(token)
    if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
        bounds = _extract_range(parsed)
        if bounds is None:
            raise ValueError(f"Invalid range token {token!r}")
        return [
            _restore_numeric_precision(bounds[0]),
            _restore_numeric_precision(bounds[1]),
        ]
    if "," in token:
        parts = [part.strip() for part in token.split(",")]
        if len(parts) == 2:
            bounds = _extract_range(parts)
            if bounds is None:
                raise ValueError(f"Invalid range token {token!r}")
            return [
                _restore_numeric_precision(bounds[0]),
                _restore_numeric_precision(bounds[1]),
            ]
    raise ValueError(
        f"Invalid range token {token!r}; expected [low,high] or low,high."
    )


_INLINE_LIMIT_OP_RE = re.compile(
    r"^\s*(?P<metric>[^\s<>=!]+)\s*(?P<op><=|>=|==|<|>)\s*(?P<rhs>.+?)\s*$"
)


def _tokenize_cli_limit_entry(raw_entry: str) -> List[str]:
    tokens = shlex.split(raw_entry)
    if len(tokens) >= 3:
        return tokens

    match = _INLINE_LIMIT_OP_RE.match(raw_entry)
    if not match:
        return tokens

    metric = match.group("metric")
    op = match.group("op")
    rhs_tokens = shlex.split(match.group("rhs"))
    return [metric, op, *rhs_tokens]


def parse_limit_cli_entry(raw_entry: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(raw_entry, dict):
        return _normalize_limit_entry_preserve_extras(raw_entry)
    if not isinstance(raw_entry, str) or not raw_entry.strip():
        raise ValueError("CLI limit entries must be non-empty strings.")

    parsed = _parse_jsonish(raw_entry)
    if isinstance(parsed, dict):
        return _normalize_limit_entry_preserve_extras(parsed)
    if parsed is not None:
        raise ValueError("CLI --limit expects a single limit object, not a list.")

    tokens = _tokenize_cli_limit_entry(raw_entry)
    if len(tokens) < 3:
        raise ValueError(
            "CLI --limit format must be 'metric <op> value' or "
            "'metric outside_range [low,high]'."
        )

    entry: Dict[str, Any] = {"metric": tokens[0], "penalize_if": tokens[1]}
    penalize_if = _normalize_penalize_if(tokens[1])

    if penalize_if in {
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "equal_to",
        "auto",
    }:
        numeric_value = _ensure_float(tokens[2])
        if numeric_value is None:
            raise ValueError(f"CLI --limit requires a numeric value, got {tokens[2]!r}")
        entry["value"] = _restore_numeric_precision(numeric_value)
    else:
        entry["range"] = _parse_cli_range_token(tokens[2])

    for token in tokens[3:]:
        if "=" not in token:
            raise ValueError(
                f"Unsupported CLI --limit token {token!r}; expected key=value."
            )
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "stat":
            entry["stat"] = value
        elif key == "enabled":
            entry["enabled"] = _parse_cli_bool(value)
        else:
            raise ValueError(
                f"Unsupported CLI --limit option {key!r}; supported extras are stat=... and enabled=..."
            )

    return _normalize_limit_entry_preserve_extras(entry)


def parse_limit_cli_entries(raw_entries: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [parse_limit_cli_entry(entry) for entry in raw_entries]


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
        ">=": "greater_than_or_equal",
        "gt": "greater_than",
        "gte": "greater_than_or_equal",
        "ge": "greater_than_or_equal",
        "greater": "greater_than",
        "greater_than": "greater_than",
        "greater_than_or_equal": "greater_than_or_equal",
        "greater_or_equal": "greater_than_or_equal",
        "above": "greater_than",
        "<": "less_than",
        "<=": "less_than_or_equal",
        "lt": "less_than",
        "lte": "less_than_or_equal",
        "le": "less_than_or_equal",
        "lower": "less_than",
        "less": "less_than",
        "less_than": "less_than",
        "less_than_or_equal": "less_than_or_equal",
        "less_or_equal": "less_than_or_equal",
        "below": "less_than",
        "==": "equal_to",
        "=": "equal_to",
        "eq": "equal_to",
        "equal": "equal_to",
        "equal_to": "equal_to",
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
    if penalize_if in {
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "equal_to",
        "auto",
    }:
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
            parsed = _parse_jsonish(stripped)
            if parsed is None:
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
