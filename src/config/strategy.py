from copy import deepcopy
from typing import Optional

from .shared_bot import BOT_SHARED_GROUPS


BOT_POSITION_SIDES = ("long", "short")
DEFAULT_STRATEGY_KIND = "trailing_grid"
EMA_ANCHOR_STRATEGY_KIND = "ema_anchor"
SUPPORTED_STRATEGY_KINDS = (DEFAULT_STRATEGY_KIND, EMA_ANCHOR_STRATEGY_KIND)

TRAILING_GRID_PARAM_KEYS = (
    "close_grid_markup_end",
    "close_grid_markup_start",
    "close_grid_qty_pct",
    "close_trailing_grid_ratio",
    "close_trailing_qty_pct",
    "close_trailing_retracement_pct",
    "close_trailing_threshold_pct",
    "ema_span_0",
    "ema_span_1",
    "entry_grid_double_down_factor",
    "entry_grid_spacing_pct",
    "entry_grid_spacing_volatility_weight",
    "entry_grid_spacing_we_weight",
    "entry_initial_ema_dist",
    "entry_initial_qty_pct",
    "entry_trailing_double_down_factor",
    "entry_trailing_grid_ratio",
    "entry_trailing_retracement_pct",
    "entry_trailing_retracement_volatility_weight",
    "entry_trailing_retracement_we_weight",
    "entry_trailing_threshold_pct",
    "entry_trailing_threshold_volatility_weight",
    "entry_trailing_threshold_we_weight",
    "entry_volatility_ema_span_hours",
)

EMA_ANCHOR_PARAM_KEYS = (
    "base_qty_pct",
    "ema_span_0",
    "ema_span_1",
    "offset",
    "offset_psize_weight",
)

STRATEGY_PARAM_KEYS_BY_KIND = {
    DEFAULT_STRATEGY_KIND: TRAILING_GRID_PARAM_KEYS,
    EMA_ANCHOR_STRATEGY_KIND: EMA_ANCHOR_PARAM_KEYS,
}

STRATEGY_DEFAULTS_BY_KIND = {
    DEFAULT_STRATEGY_KIND: {
        "long": {
            "close_grid_markup_end": 0.0094,
            "close_grid_markup_start": 0.00634,
            "close_grid_qty_pct": 0.51,
            "close_trailing_grid_ratio": -0.76,
            "close_trailing_qty_pct": 0.05,
            "close_trailing_retracement_pct": 0.00279,
            "close_trailing_threshold_pct": 0.001,
            "ema_span_0": 770,
            "ema_span_1": 210,
            "entry_grid_double_down_factor": 0.73,
            "entry_grid_spacing_pct": 0.033,
            "entry_grid_spacing_volatility_weight": 2.4,
            "entry_grid_spacing_we_weight": 0.135,
            "entry_initial_ema_dist": 0.0097,
            "entry_initial_qty_pct": 0.0276,
            "entry_trailing_double_down_factor": 0.9,
            "entry_trailing_grid_ratio": -0.5,
            "entry_trailing_retracement_pct": 0.0276,
            "entry_trailing_retracement_volatility_weight": 87,
            "entry_trailing_retracement_we_weight": 3.97,
            "entry_trailing_threshold_pct": 0.0029,
            "entry_trailing_threshold_volatility_weight": 76,
            "entry_trailing_threshold_we_weight": 1.31,
            "entry_volatility_ema_span_hours": 1690,
        },
        "short": {
            "close_grid_markup_end": 0.0015,
            "close_grid_markup_start": 0.0015,
            "close_grid_qty_pct": 0.05,
            "close_trailing_grid_ratio": -1,
            "close_trailing_qty_pct": 0.05,
            "close_trailing_retracement_pct": 0.001,
            "close_trailing_threshold_pct": 0.001,
            "ema_span_0": 100,
            "ema_span_1": 100,
            "entry_grid_double_down_factor": 0.5,
            "entry_grid_spacing_pct": 0.025,
            "entry_grid_spacing_volatility_weight": 1,
            "entry_grid_spacing_we_weight": 0,
            "entry_initial_ema_dist": -0.01,
            "entry_initial_qty_pct": 0.01,
            "entry_trailing_double_down_factor": 0.5,
            "entry_trailing_grid_ratio": -0.5,
            "entry_trailing_retracement_pct": 0.001,
            "entry_trailing_retracement_volatility_weight": 1,
            "entry_trailing_retracement_we_weight": 0,
            "entry_trailing_threshold_pct": 0.001,
            "entry_trailing_threshold_volatility_weight": 1,
            "entry_trailing_threshold_we_weight": 0,
            "entry_volatility_ema_span_hours": 672,
        },
    },
    EMA_ANCHOR_STRATEGY_KIND: {
        "long": {
            "base_qty_pct": 0.01,
            "ema_span_0": 200.0,
            "ema_span_1": 800.0,
            "offset": 0.002,
            "offset_psize_weight": 0.1,
        },
        "short": {
            "base_qty_pct": 0.01,
            "ema_span_0": 200.0,
            "ema_span_1": 800.0,
            "offset": 0.002,
            "offset_psize_weight": 0.1,
        },
    },
}


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

    result = {}
    for key in strategy_keys:
        if isinstance(override_side, dict) and key in override_side:
            result[key] = deepcopy(override_side[key])
            continue
        if isinstance(strategy_side, dict) and key in strategy_side:
            result[key] = deepcopy(strategy_side[key])
            continue
        if key in side_defaults:
            result[key] = deepcopy(side_defaults[key])
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
                if key in current_strategy_side:
                    continue
                if kind == normalized_kind and key in defaults_by_side:
                    current_strategy_side[key] = deepcopy(defaults_by_side[key])
                    if tracker is not None:
                        tracker.add(["bot", pside, "strategy", kind, key], current_strategy_side[key])
                elif kind != normalized_kind and key in defaults_by_side:
                    current_strategy_side[key] = deepcopy(defaults_by_side[key])
                    if tracker is not None:
                        tracker.add(["bot", pside, "strategy", kind, key], current_strategy_side[key])


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
