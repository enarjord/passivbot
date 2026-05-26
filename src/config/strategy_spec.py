from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any

import passivbot_rust as pbr


BOT_POSITION_SIDES = ("long", "short")
DEFAULT_STRATEGY_KIND = "trailing_martingale"
EMA_ANCHOR_STRATEGY_KIND = "ema_anchor"


def _strategy_metadata_error(exc: Exception) -> RuntimeError:
    return RuntimeError(
        "Rust strategy metadata is unavailable or stale. Rebuild the Rust extension with "
        "`cd passivbot-rust && maturin develop --release && cd ..`, then retry."
    )


def _set_path(mapping: dict, path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for part in path[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[path[-1]] = deepcopy(value)


@lru_cache(maxsize=1)
def get_supported_strategy_kinds() -> tuple[str, ...]:
    try:
        kinds = pbr.get_strategy_kinds()
    except AttributeError as exc:
        raise _strategy_metadata_error(exc) from exc
    except Exception as exc:
        raise _strategy_metadata_error(exc) from exc
    normalized = tuple(str(kind).strip().lower() for kind in kinds if str(kind).strip())
    if not normalized:
        raise RuntimeError("Rust strategy metadata returned no supported strategy kinds")
    return normalized


def normalize_strategy_kind(value) -> str:
    kind = str(value or DEFAULT_STRATEGY_KIND).strip().lower()
    normalized = kind or DEFAULT_STRATEGY_KIND
    supported = get_supported_strategy_kinds()
    if normalized not in supported:
        allowed = ", ".join(sorted(supported))
        raise ValueError(f"unsupported strategy kind {normalized!r}; expected one of {{{allowed}}}")
    return normalized


@lru_cache(maxsize=None)
def get_strategy_spec(strategy_kind: str = DEFAULT_STRATEGY_KIND) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    try:
        spec = pbr.get_strategy_spec(normalized_kind)
    except AttributeError as exc:
        raise _strategy_metadata_error(exc) from exc
    except ValueError:
        raise
    except Exception as exc:
        raise _strategy_metadata_error(exc) from exc
    if not isinstance(spec, dict):
        raise RuntimeError(
            f"Rust strategy metadata for {normalized_kind!r} must be a dict; "
            f"got {type(spec).__name__}"
        )
    return spec


def _strategy_leaf_path(param: dict) -> tuple[str, ...] | None:
    config_path = tuple(param.get("config_path", ()))
    if len(config_path) < 3 or config_path[0] != "strategy" or config_path[1] not in BOT_POSITION_SIDES:
        return None
    return tuple(str(part) for part in config_path[2:])


def get_strategy_param_keys(strategy_kind: str) -> tuple[str, ...]:
    spec = get_strategy_spec(strategy_kind)
    keys: list[str] = []
    for param in spec.get("parameters", []):
        if not isinstance(param, dict):
            continue
        leaf_path = _strategy_leaf_path(param)
        if leaf_path is not None:
            keys.append(".".join(leaf_path))
    return tuple(dict.fromkeys(keys))


def get_strategy_defaults(strategy_kind: str) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    spec = get_strategy_spec(normalized_kind)
    result = {pside: {} for pside in BOT_POSITION_SIDES}
    for param in spec.get("parameters", []):
        if not isinstance(param, dict):
            continue
        pside = param.get("side")
        leaf_path = _strategy_leaf_path(param)
        if pside not in BOT_POSITION_SIDES or leaf_path is None:
            continue
        if "default" not in param:
            raise RuntimeError(
                f"Rust strategy metadata for {normalized_kind!r} is missing default for "
                f"{param.get('optimize_key')!r}"
            )
        _set_path(result[pside], leaf_path, param["default"])
    return result


def get_all_strategy_defaults() -> dict:
    defaults_by_kind = {
        kind: get_strategy_defaults(kind)
        for kind in get_supported_strategy_kinds()
    }
    return {
        pside: {kind: deepcopy(defaults[pside]) for kind, defaults in defaults_by_kind.items()}
        for pside in BOT_POSITION_SIDES
    }


def get_strategy_optimize_bounds(strategy_kind: str) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    spec = get_strategy_spec(normalized_kind)
    result = {pside: {} for pside in BOT_POSITION_SIDES}
    for param in spec.get("parameters", []):
        if not isinstance(param, dict):
            continue
        pside = param.get("side")
        leaf_path = _strategy_leaf_path(param)
        if pside not in BOT_POSITION_SIDES or leaf_path is None:
            continue
        if "bounds" not in param:
            raise RuntimeError(
                f"Rust strategy metadata for {normalized_kind!r} is missing bounds for "
                f"{param.get('optimize_key')!r}"
            )
        _set_path(result[pside], leaf_path, list(param["bounds"]))
    return result


def get_all_strategy_optimize_bounds() -> dict:
    bounds_by_kind = {
        kind: get_strategy_optimize_bounds(kind)
        for kind in get_supported_strategy_kinds()
    }
    return {
        pside: {kind: deepcopy(bounds[pside]) for kind, bounds in bounds_by_kind.items()}
        for pside in BOT_POSITION_SIDES
    }


def strategy_optimize_key_path_map(strategy_kind: str) -> dict[str, tuple[str, ...]]:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    spec = get_strategy_spec(normalized_kind)
    mapping: dict[str, tuple[str, ...]] = {}
    for param in spec.get("parameters", []):
        if not isinstance(param, dict):
            continue
        optimize_key = param.get("optimize_key")
        config_path = tuple(param.get("config_path", ()))
        if not isinstance(optimize_key, str):
            continue
        if len(config_path) >= 3 and config_path[0] == "strategy" and config_path[1] in BOT_POSITION_SIDES:
            mapping[optimize_key] = (
                "bot",
                str(config_path[1]),
                "strategy",
                normalized_kind,
                *(str(part) for part in config_path[2:]),
            )
    return mapping


def strategy_field_names_by_side(strategy_kind: str) -> dict[str, set[str]]:
    spec = get_strategy_spec(strategy_kind)
    fields = {pside: set() for pside in BOT_POSITION_SIDES}
    for param in spec.get("parameters", []):
        if not isinstance(param, dict):
            continue
        side = param.get("side")
        name = param.get("name")
        if side in fields and isinstance(name, str):
            fields[side].add(name)
    return fields
