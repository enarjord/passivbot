"""Explicit released-v8 ownership conversion for shared bot policy."""

from copy import deepcopy


LIVE_POLICY_PATHS = {
    "strategy_kind": "bot.global.strategy_kind",
    "hedge_mode": "bot.global.hedge_mode",
    "minimum_coin_age_days": "bot.global.universe.minimum_coin_age_days",
    "filter_by_min_effective_cost": "bot.global.universe.filter_by_min_effective_cost",
    "market_orders_allowed": "bot.global.execution.market_orders_allowed",
    "market_order_near_touch_threshold": "bot.global.execution.market_order_near_touch_threshold",
    "forager_score_hysteresis_pct": "bot.global.forager.score_hysteresis_pct",
    "warmup_ratio": "bot.global.warmup.ratio",
    "max_warmup_minutes": "bot.global.warmup.max_minutes",
}
COIN_SOURCE_FIELDS = ("approved_coins", "ignored_coins")


def shared_policy_selector(path):
    """Retarget exact released selectors; runtime paths have no alias fallback."""
    for field, target in LIVE_POLICY_PATHS.items():
        old = f"live.{field}"
        if path == old or path.startswith(old + "."):
            return target + path[len(old) :]
    for field in COIN_SOURCE_FIELDS:
        old = f"live.{field}"
        if path == old:
            return f"bot.*.universe.{field}"
        for side in ("long", "short", "*"):
            if path == f"{old}.{side}":
                return f"bot.{side}.universe.{field}"
    return path


def forced_normal_coins(coin_overrides):
    result = {side: [] for side in ("long", "short")}
    for coin, patch in (coin_overrides or {}).items():
        if not isinstance(patch, dict):
            continue
        live = patch.get("live", {})
        if not isinstance(live, dict):
            continue
        for side in result:
            mode = live.get(f"forced_mode_{side}")
            if isinstance(mode, str) and mode.strip().lower() in {"normal", "n"}:
                result[side].append(coin)
    return result


def move_shared_policy(document, *, record):
    """Move authored leaves without hydrating sparse coin/scenario documents."""
    live = document.get("live", {})
    if not isinstance(live, dict):
        return

    def put(path, value, source):
        node = document
        for part in path.split(".")[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{source} conflicts with {path}")
        leaf = path.split(".")[-1]
        if leaf in node and node[leaf] != value:
            raise ValueError(f"{source} conflicts with {path}")
        node[leaf] = deepcopy(value)
        record(
            source,
            value,
            {path: value},
            "shared live/backtest policy moved to bot ownership",
        )

    for field, path in LIVE_POLICY_PATHS.items():
        if field in live:
            put(path, live.pop(field), f"live.{field}")
    for field in COIN_SOURCE_FIELDS:
        if field not in live:
            continue
        value = live.pop(field)
        if isinstance(value, dict):
            if set(value) - {"long", "short"}:
                raise ValueError(f"live.{field} must use only long/short source keys")
            sources = value
        else:
            sources = {side: value for side in ("long", "short")}
        for side, source in sources.items():
            put(f"bot.{side}.universe.{field}", source, f"live.{field}.{side}")


def preserve_forced_normal_selection(document, *, record):
    """Retain operational modes and expose their selection intent once coins are merged."""
    for side, coins in forced_normal_coins(document.get("coin_overrides")).items():
        if not coins:
            continue
        path = f"bot.{side}.forager.always_include"
        forager = (
            document.setdefault("bot", {})
            .setdefault(side, {})
            .setdefault("forager", {})
        )
        old = deepcopy(forager.get("always_include", []))
        value = list(dict.fromkeys([*old, *coins]))
        forager["always_include"] = value
        record(
            f"coin_overrides.*.live.forced_mode_{side}=normal",
            old,
            {path: value},
            "preserved released coin-normal forced selection and operational exemptions",
        )
