from copy import deepcopy
from typing import Optional

from .shared_bot import BOT_SHARED_GROUPS


BOT_POSITION_SIDES = ("long", "short")
DEFAULT_STRATEGY_KIND = "trailing_martingale"
EMA_ANCHOR_STRATEGY_KIND = "ema_anchor"
SUPPORTED_STRATEGY_KINDS = (DEFAULT_STRATEGY_KIND, EMA_ANCHOR_STRATEGY_KIND)

TRAILING_MARTINGALE_PARAM_KEYS = (
    "ema_span_0",
    "ema_span_1",
    "volatility_ema_span_1h",
    "volatility_ema_span_1m",
    "entry.double_down_factor",
    "entry.initial_ema_dist",
    "entry.initial_qty_pct",
    "entry.threshold_base_pct",
    "entry.threshold_we_weight",
    "entry.threshold_volatility_1h_weight",
    "entry.threshold_volatility_1m_weight",
    "entry.retracement_base_pct",
    "entry.retracement_we_weight",
    "entry.retracement_volatility_1h_weight",
    "entry.retracement_volatility_1m_weight",
    "close.qty_pct",
    "close.threshold_base_pct",
    "close.threshold_we_weight",
    "close.threshold_volatility_1h_weight",
    "close.threshold_volatility_1m_weight",
    "close.retracement_base_pct",
    "close.retracement_volatility_1h_weight",
    "close.retracement_volatility_1m_weight",
)

EMA_ANCHOR_PARAM_KEYS = (
    "base_qty_pct",
    "ema_span_0",
    "ema_span_1",
    "entry_double_down_factor",
    "offset",
    "offset_volatility_ema_span_1m",
    "offset_volatility_1m_weight",
    "entry_volatility_ema_span_1h",
    "offset_volatility_1h_weight",
    "offset_psize_weight",
)

STRATEGY_PARAM_KEYS_BY_KIND = {
    DEFAULT_STRATEGY_KIND: TRAILING_MARTINGALE_PARAM_KEYS,
    EMA_ANCHOR_STRATEGY_KIND: EMA_ANCHOR_PARAM_KEYS,
}

STRATEGY_DEFAULTS_BY_KIND = {
    DEFAULT_STRATEGY_KIND: {
        "long": {
            "ema_span_0": 770,
            "ema_span_1": 210,
            "volatility_ema_span_1h": 1690,
            "volatility_ema_span_1m": 60.0,
            "entry": {
                "double_down_factor": 0.73,
                "initial_ema_dist": 0.0097,
                "initial_qty_pct": 0.0276,
                "threshold_base_pct": 0.033,
                "threshold_we_weight": 0.135,
                "threshold_volatility_1h_weight": 2.4,
                "threshold_volatility_1m_weight": 0.0,
                "retracement_base_pct": 0.0,
                "retracement_we_weight": 0.0,
                "retracement_volatility_1h_weight": 0.0,
                "retracement_volatility_1m_weight": 0.0,
            },
            "close": {
                "qty_pct": 0.1,
                "threshold_base_pct": 0.006,
                "threshold_we_weight": -0.004,
                "threshold_volatility_1h_weight": 1.0,
                "threshold_volatility_1m_weight": 0.0,
                "retracement_base_pct": 0.0,
                "retracement_volatility_1h_weight": 0.0,
                "retracement_volatility_1m_weight": 0.0,
            },
        },
        "short": {
            "ema_span_0": 100,
            "ema_span_1": 100,
            "volatility_ema_span_1h": 672,
            "volatility_ema_span_1m": 60.0,
            "entry": {
                "double_down_factor": 0.5,
                "initial_ema_dist": -0.01,
                "initial_qty_pct": 0.01,
                "threshold_base_pct": 0.025,
                "threshold_we_weight": 0.0,
                "threshold_volatility_1h_weight": 1.0,
                "threshold_volatility_1m_weight": 0.0,
                "retracement_base_pct": 0.0,
                "retracement_we_weight": 0.0,
                "retracement_volatility_1h_weight": 0.0,
                "retracement_volatility_1m_weight": 0.0,
            },
            "close": {
                "qty_pct": 0.1,
                "threshold_base_pct": 0.006,
                "threshold_we_weight": -0.004,
                "threshold_volatility_1h_weight": 1.0,
                "threshold_volatility_1m_weight": 0.0,
                "retracement_base_pct": 0.0,
                "retracement_volatility_1h_weight": 0.0,
                "retracement_volatility_1m_weight": 0.0,
            },
        },
    },
    EMA_ANCHOR_STRATEGY_KIND: {
        "long": {
            "base_qty_pct": 0.01,
            "ema_span_0": 200.0,
            "ema_span_1": 800.0,
            "entry_double_down_factor": 0.0,
            "offset": 0.002,
            "offset_volatility_ema_span_1m": 60.0,
            "offset_volatility_1m_weight": 0.0,
            "entry_volatility_ema_span_1h": 24.0,
            "offset_volatility_1h_weight": 0.0,
            "offset_psize_weight": 0.1,
        },
        "short": {
            "base_qty_pct": 0.01,
            "ema_span_0": 200.0,
            "ema_span_1": 800.0,
            "entry_double_down_factor": 0.0,
            "offset": 0.002,
            "offset_volatility_ema_span_1m": 60.0,
            "offset_volatility_1m_weight": 0.0,
            "entry_volatility_ema_span_1h": 24.0,
            "offset_volatility_1h_weight": 0.0,
            "offset_psize_weight": 0.1,
        },
    },
}


def _normalize_strategy_side_value(key: str, value, *, strategy_kind: str, pside: str):
    return value


def _path_parts(key: str) -> tuple[str, ...]:
    return tuple(part for part in key.split(".") if part)


def _get_path(mapping: dict, key: str):
    current = mapping
    for part in _path_parts(key):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _has_path(mapping: dict, key: str) -> bool:
    sentinel = object()
    return _get_path_or(mapping, key, sentinel) is not sentinel


def _get_path_or(mapping: dict, key: str, default):
    current = mapping
    for part in _path_parts(key):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _set_path(mapping: dict, key: str, value) -> None:
    current = mapping
    parts = _path_parts(key)
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = deepcopy(value)


def _iter_leaf_paths(mapping: dict, prefix: tuple[str, ...] = ()):
    for key, value in mapping.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _iter_leaf_paths(value, path)
        else:
            yield ".".join(path), value


def get_all_strategy_defaults() -> dict:
    return {
        pside: {kind: deepcopy(defaults[pside]) for kind, defaults in STRATEGY_DEFAULTS_BY_KIND.items()}
        for pside in BOT_POSITION_SIDES
    }


def normalize_strategy_kind(value) -> str:
    kind = str(value or DEFAULT_STRATEGY_KIND).strip().lower()
    return kind or DEFAULT_STRATEGY_KIND


def get_strategy_param_keys(strategy_kind: str) -> tuple[str, ...]:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    try:
        return STRATEGY_PARAM_KEYS_BY_KIND[normalized_kind]
    except KeyError as exc:
        allowed = ", ".join(sorted(STRATEGY_PARAM_KEYS_BY_KIND))
        raise ValueError(f"unsupported strategy kind {normalized_kind!r}; expected one of {{{allowed}}}") from exc


def get_strategy_defaults(strategy_kind: str) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    try:
        return deepcopy(STRATEGY_DEFAULTS_BY_KIND[normalized_kind])
    except KeyError as exc:
        allowed = ", ".join(sorted(STRATEGY_DEFAULTS_BY_KIND))
        raise ValueError(f"unsupported strategy kind {normalized_kind!r}; expected one of {{{allowed}}}") from exc


def get_strategy_store(bot_side: dict | None) -> dict:
    if not isinstance(bot_side, dict):
        return {}
    strategy_store = bot_side.get("strategy")
    if isinstance(strategy_store, dict):
        return strategy_store
    return {}


def get_active_strategy_side(
    bot_side: dict | None,
    *,
    strategy_kind: str = DEFAULT_STRATEGY_KIND,
    pside: str | None = None,
) -> dict:
    del pside
    normalized_kind = normalize_strategy_kind(strategy_kind)
    strategy_store = get_strategy_store(bot_side)
    active = strategy_store.get(normalized_kind)
    if isinstance(active, dict):
        return active
    return {}


def get_active_strategy_config(config: dict, *, strategy_kind: str | None = None) -> dict:
    normalized_kind = normalize_strategy_kind(
        strategy_kind if strategy_kind is not None else config.get("live", {}).get("strategy_kind")
    )
    bot_cfg = config.get("bot", {})
    return {
        pside: deepcopy(
            get_active_strategy_side(bot_cfg.get(pside, {}), strategy_kind=normalized_kind, pside=pside)
        )
        for pside in BOT_POSITION_SIDES
    }


def build_runtime_strategy_side(
    strategy_side: dict | None = None,
    *,
    strategy_kind: str = DEFAULT_STRATEGY_KIND,
    pside: str | None = None,
    override_side: dict | None = None,
) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    strategy_keys = get_strategy_param_keys(normalized_kind)
    defaults_by_side = get_strategy_defaults(normalized_kind)
    side_defaults = (
        defaults_by_side.get(pside, {})
        if pside in BOT_POSITION_SIDES
        else {}
    )
    if isinstance(strategy_side, dict) and "strategy" in strategy_side:
        strategy_side = get_active_strategy_side(
            strategy_side,
            strategy_kind=normalized_kind,
            pside=pside,
        )
    if isinstance(override_side, dict) and "strategy" in override_side:
        override_side = get_active_strategy_side(
            override_side,
            strategy_kind=normalized_kind,
            pside=pside,
        )

    result = deepcopy(side_defaults)
    for key in strategy_keys:
        if isinstance(override_side, dict) and _has_path(override_side, key):
            _set_path(
                result,
                key,
                _normalize_strategy_side_value(
                    key,
                    _get_path(override_side, key),
                    strategy_kind=normalized_kind,
                    pside=pside or "",
                ),
            )
            continue
        if isinstance(strategy_side, dict) and _has_path(strategy_side, key):
            _set_path(
                result,
                key,
                _normalize_strategy_side_value(
                    key,
                    _get_path(strategy_side, key),
                    strategy_kind=normalized_kind,
                    pside=pside or "",
                ),
            )
            continue
    return result


def sync_canonical_strategy_config(config: dict, *, tracker: Optional[object] = None) -> None:
    live_cfg = config.setdefault("live", {})
    normalized_kind = normalize_strategy_kind(live_cfg.get("strategy_kind"))
    if "strategy_kind" not in live_cfg:
        live_cfg["strategy_kind"] = normalized_kind
        if tracker is not None:
            tracker.add(["live", "strategy_kind"], normalized_kind)
    elif live_cfg["strategy_kind"] != normalized_kind:
        if tracker is not None:
            tracker.update(["live", "strategy_kind"], live_cfg["strategy_kind"], normalized_kind)
        live_cfg["strategy_kind"] = normalized_kind

    bot_cfg = config.setdefault("bot", {})
    for pside in BOT_POSITION_SIDES:
        bot_side = bot_cfg.setdefault(pside, {})
        if not isinstance(bot_side, dict):
            raise TypeError(f"config.bot.{pside} must be a dict; got {type(bot_side).__name__}")
        strategy_store = bot_side.get("strategy")
        if strategy_store is None:
            strategy_store = {}
            bot_side["strategy"] = strategy_store
            if tracker is not None:
                tracker.add(["bot", pside, "strategy"], {})
        elif not isinstance(strategy_store, dict):
            raise TypeError(
                f"config.bot.{pside}.strategy must be a dict; got {type(strategy_store).__name__}"
            )

        for kind, defaults_by_side in get_all_strategy_defaults()[pside].items():
            current_strategy_side = strategy_store.get(kind)
            if current_strategy_side is None:
                current_strategy_side = {}
                strategy_store[kind] = current_strategy_side
                if tracker is not None:
                    tracker.add(["bot", pside, "strategy", kind], {})
            elif not isinstance(current_strategy_side, dict):
                raise TypeError(
                    f"config.bot.{pside}.strategy.{kind} must be a dict; "
                    f"got {type(current_strategy_side).__name__}"
                )

            for key in get_strategy_param_keys(kind):
                if key in bot_side:
                    flat_value = bot_side.pop(key)
                    if _get_path_or(current_strategy_side, key, None) != flat_value:
                        old_value = _get_path_or(current_strategy_side, key, None)
                        _set_path(current_strategy_side, key, flat_value)
                        if tracker is not None:
                            if old_value is None:
                                tracker.rename(
                                    ["bot", pside, key],
                                    ["bot", pside, "strategy", kind, *_path_parts(key)],
                                    flat_value,
                                )
                            else:
                                tracker.update(
                                    ["bot", pside, "strategy", kind, *_path_parts(key)],
                                    old_value,
                                    flat_value,
                                )
                    elif tracker is not None:
                        tracker.remove(["bot", pside, key], flat_value)
                if not _has_path(current_strategy_side, key):
                    if _has_path(defaults_by_side, key):
                        _set_path(current_strategy_side, key, _get_path(defaults_by_side, key))
                        if tracker is not None:
                            tracker.add(
                                ["bot", pside, "strategy", kind, *_path_parts(key)],
                                _get_path(current_strategy_side, key),
                            )
                normalized_value = _normalize_strategy_side_value(
                    key,
                    _get_path(current_strategy_side, key),
                    strategy_kind=kind,
                    pside=pside,
                )
                if _get_path(current_strategy_side, key) != normalized_value:
                    if tracker is not None:
                        tracker.update(
                            ["bot", pside, "strategy", kind, *_path_parts(key)],
                            _get_path(current_strategy_side, key),
                            normalized_value,
                        )
                    _set_path(current_strategy_side, key, normalized_value)


def prune_inactive_strategy_subtrees(config: dict, *, tracker: Optional[object] = None) -> None:
    bot_cfg = config.get("bot")
    if not isinstance(bot_cfg, dict):
        return
    active_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    for pside in BOT_POSITION_SIDES:
        bot_side = bot_cfg.get(pside)
        if not isinstance(bot_side, dict):
            continue
        strategy_store = bot_side.get("strategy")
        if not isinstance(strategy_store, dict):
            continue
        for kind in list(strategy_store):
            if kind == active_kind:
                continue
            removed = strategy_store.pop(kind)
            if tracker is not None:
                tracker.remove(["bot", pside, "strategy", kind], removed)


def merge_runtime_bot_side(
    bot_side: dict,
    strategy_side: dict | None = None,
    *,
    pside: str | None = None,
    override_side: dict | None = None,
    strategy_kind: str = DEFAULT_STRATEGY_KIND,
) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    strategy_keys = set(get_strategy_param_keys(normalized_kind))
    merged = deepcopy(bot_side) if isinstance(bot_side, dict) else {}
    merged.pop("strategy", None)
    for group_name in BOT_SHARED_GROUPS:
        merged.pop(group_name, None)
    for key in list(merged):
        if key in strategy_keys:
            merged.pop(key)
    if isinstance(override_side, dict):
        for key, value in override_side.items():
            if key == "strategy":
                continue
            if key in strategy_keys:
                continue
            merged[key] = deepcopy(value)
    return merged
