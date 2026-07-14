from __future__ import annotations

from typing import Iterable

from .shared_bot import canonical_shared_bot_path_for_flat_key, resolve_shared_bot_path
from .strategy_spec import (
    BOT_POSITION_SIDES,
    get_strategy_param_keys,
    get_supported_strategy_kinds,
    normalize_strategy_kind,
    strategy_optimize_key_path_map,
)


OPTIMIZABLE_BOT_KEY_PATHS = {
    "long_forager_volume_ema_span_1m": ("bot", "long", "forager", "volume_ema_span_1m"),
    "long_filter_volume_ema_span_1m": ("bot", "long", "forager", "volume_ema_span_1m"),
    "long_forager_volatility_ema_span_1m": ("bot", "long", "forager", "volatility_ema_span_1m"),
    "long_filter_volatility_ema_span_1m": ("bot", "long", "forager", "volatility_ema_span_1m"),
    "long_forager_score_weights_volume": ("bot", "long", "forager", "score_weights", "volume"),
    "long_forager_score_weights_ema_readiness": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "ema_readiness",
    ),
    "long_forager_score_weights_volatility": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "volatility",
    ),
    "short_forager_score_weights_volume": ("bot", "short", "forager", "score_weights", "volume"),
    "short_forager_score_weights_ema_readiness": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "ema_readiness",
    ),
    "short_forager_score_weights_volatility": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "volatility",
    ),
    "short_forager_volume_ema_span_1m": ("bot", "short", "forager", "volume_ema_span_1m"),
    "short_filter_volume_ema_span_1m": ("bot", "short", "forager", "volume_ema_span_1m"),
    "short_forager_volatility_ema_span_1m": ("bot", "short", "forager", "volatility_ema_span_1m"),
    "short_filter_volatility_ema_span_1m": ("bot", "short", "forager", "volatility_ema_span_1m"),
}

DEPRECATED_OPTIMIZE_BOUND_ALIASES = {
    "long_filter_volume_ema_span_1m": "long_forager_volume_ema_span_1m",
    "long_filter_volatility_ema_span_1m": "long_forager_volatility_ema_span_1m",
    "short_filter_volume_ema_span_1m": "short_forager_volume_ema_span_1m",
    "short_filter_volatility_ema_span_1m": "short_forager_volatility_ema_span_1m",
}


def canonical_optimizer_key(key: str) -> str:
    return DEPRECATED_OPTIMIZE_BOUND_ALIASES.get(key, key)


def active_strategy_kind(config: dict) -> str:
    live_cfg = config.get("live")
    raw_kind = live_cfg.get("strategy_kind") if isinstance(live_cfg, dict) else None
    return normalize_strategy_kind(raw_kind)


def _strategy_path_map_for_config(config: dict) -> dict[str, tuple[str, ...]]:
    return strategy_optimize_key_path_map(active_strategy_kind(config))


def resolve_optimizer_key_path(config: dict, key: str) -> tuple[str, ...] | None:
    canonical_key = canonical_optimizer_key(key)
    strategy_path_map = _strategy_path_map_for_config(config)
    if canonical_key in strategy_path_map:
        return strategy_path_map[canonical_key]
    if canonical_key in OPTIMIZABLE_BOT_KEY_PATHS:
        return OPTIMIZABLE_BOT_KEY_PATHS[canonical_key]
    try:
        pside, flat_key = canonical_key.split("_", 1)
    except ValueError:
        return None
    if pside not in BOT_POSITION_SIDES:
        return None
    bot_side = config.get("bot", {}).get(pside, {})
    return resolve_shared_bot_path(bot_side, pside, flat_key) or ("bot", pside, flat_key)


def canonical_path_for_bot_side_flat_key(
    config: dict,
    pside: str,
    flat_key: str,
) -> tuple[str, ...] | None:
    if pside not in BOT_POSITION_SIDES:
        return None
    shared_path = canonical_shared_bot_path_for_flat_key(pside, flat_key)
    if shared_path is not None:
        return shared_path
    strategy_kind = active_strategy_kind(config)
    strategy_param_keys = set(get_strategy_param_keys(strategy_kind))
    if flat_key in strategy_param_keys:
        return ("bot", pside, "strategy", strategy_kind, *tuple(flat_key.split(".")))
    optimizer_path = resolve_optimizer_key_path(config, f"{pside}_{flat_key}")
    if optimizer_path is not None and len(optimizer_path) >= 4 and optimizer_path[:3] == (
        "bot",
        pside,
        "strategy",
    ):
        return optimizer_path
    return None


def resolve_dotted_config_path(config: dict, selector_or_path: str) -> tuple[str, ...] | None:
    raw_parts = tuple(part.strip() for part in selector_or_path.split(".") if part.strip())
    if not raw_parts:
        return ("",) if selector_or_path == "" else None
    if raw_parts[0] in BOT_POSITION_SIDES or (
        raw_parts[0] == "*"
        and len(raw_parts) >= 2
        and raw_parts[1] in ("strategy", "risk", "forager", "hsl", "unstuck")
    ):
        parts = ("bot", *raw_parts)
    else:
        parts = raw_parts
    if len(parts) == 3 and parts[0] == "bot" and parts[1] in BOT_POSITION_SIDES:
        canonical_path = canonical_path_for_bot_side_flat_key(config, parts[1], parts[2])
        if canonical_path is not None:
            return canonical_path
    if (
        len(parts) >= 4
        and parts[0] == "bot"
        and parts[1] in (*BOT_POSITION_SIDES, "*")
        and parts[2] == "strategy"
        and parts[3] not in get_supported_strategy_kinds()
        and parts[3] != "*"
    ):
        parts = (*parts[:3], active_strategy_kind(config), *parts[3:])
    return tuple(parts)


def require_existing_config_path(config: dict, selector_or_path: str) -> tuple[str, ...]:
    resolved = resolve_dotted_config_path(config, selector_or_path)
    if resolved is None or not resolved or any(part == "" for part in resolved):
        raise ValueError("Override paths must not be empty")

    target = config
    traversed: list[str] = []
    for part in resolved[:-1]:
        traversed.append(part)
        dotted = ".".join(traversed)
        if not isinstance(target, dict) or part not in target:
            raise KeyError(
                f"Unknown override path {selector_or_path!r}: missing {dotted}"
            )
        target = target[part]
        if not isinstance(target, dict):
            raise KeyError(
                f"Unknown override path {selector_or_path!r}: {dotted} is not a mapping"
            )

    final_key = resolved[-1]
    full_path = ".".join(resolved)
    if not isinstance(target, dict) or final_key not in target:
        raise KeyError(
            f"Unknown override path {selector_or_path!r}: missing {full_path}"
        )
    return resolved


def path_matches_selector(path: tuple[str, ...], selector_path: tuple[str, ...]) -> bool:
    if len(selector_path) > len(path):
        return False
    return all(
        selector_part == "*" or selector_part == path_part
        for selector_part, path_part in zip(selector_path, path)
    )


def path_suffix_matches_selector(path: tuple[str, ...], selector_path: tuple[str, ...]) -> bool:
    if len(selector_path) > len(path):
        return False
    suffix = path[-len(selector_path) :] if selector_path else ()
    return all(
        selector_part == "*" or selector_part == path_part
        for selector_part, path_part in zip(selector_path, suffix)
    )


def bound_path_matches_selector(path: tuple[str, ...], selector_path: tuple[str, ...]) -> bool:
    return path_matches_selector(path, selector_path) or path_suffix_matches_selector(
        path, selector_path
    )


def resolve_bound_selectors(
    config: dict,
    selectors: Iterable[str],
    flat_bounds: dict,
) -> dict[str, tuple[str, ...]]:
    resolved: dict[str, tuple[str, ...]] = {}
    bound_paths = {
        key: path
        for key in flat_bounds
        if isinstance(key, str) and (path := resolve_optimizer_key_path(config, key)) is not None
    }
    for selector in selectors:
        selector_path = resolve_dotted_config_path(config, str(selector))
        if selector_path is None:
            continue
        for key, path in bound_paths.items():
            if bound_path_matches_selector(path, selector_path):
                resolved[key] = path
    return resolved
