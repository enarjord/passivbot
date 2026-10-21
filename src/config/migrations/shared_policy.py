"""Explicit released-v8 ownership conversion for shared bot policy."""

from copy import deepcopy


LIVE_POLICY_PATHS = {
    "strategy_kind": "bot.global.strategy_kind",
    "hedge_mode": "bot.global.hedge_mode",
    "market_orders_allowed": "bot.global.execution.market_orders_allowed",
    "market_order_near_touch_threshold": "bot.global.execution.market_order_near_touch_fraction",
    "forager_score_hysteresis_pct": "bot.global.forager.score_hysteresis_margin",
    "warmup_ratio": "bot.global.warmup.span_multiplier",
    "max_warmup_minutes": "bot.global.warmup.max_minutes",
}
MINIMUM_COST_SOURCE_PATHS = {"live.filter_by_min_effective_cost"}


def minimum_cost_policy_targets(path):
    """Only the explicit released converter fans a shared policy out to both sides."""
    if path in MINIMUM_COST_SOURCE_PATHS:
        return tuple(f"bot.{side}.universe.filter_by_min_effective_cost" for side in ("long", "short"))
    return ()


COIN_SOURCE_FIELDS = ("approved_coins", "ignored_coins")


def shared_policy_selector(path):
    """Retarget exact released selectors; runtime paths have no alias fallback."""
    if path == "live.minimum_coin_age_days":
        return "bot.*.universe.minimum_coin_age_days"
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
            mode = live.get("modes", {}).get(side)
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

    if "minimum_coin_age_days" in live:
        value = live.pop("minimum_coin_age_days")
        for side in ("long", "short"):
            put(f"bot.{side}.universe.minimum_coin_age_days", value, "live.minimum_coin_age_days")
        record("live.minimum_coin_age_days", value,
               {f"bot.{side}.universe.minimum_coin_age_days": value for side in ("long", "short")},
               "age now gates initial entries in every live/backtest selection mode; pre-eligibility history remains available for warmup")
    sources = []
    if "filter_by_min_effective_cost" in live:
        sources.append(("live.filter_by_min_effective_cost", live.pop("filter_by_min_effective_cost")))
    for source, value in sources:
        if type(value) is not bool:
            raise TypeError(f"{source} must be a boolean")
        for target in minimum_cost_policy_targets(source):
            put(target, value, source)
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


def _overlay_coin_overrides(base, patch):
    """Apply the released replacement semantics to coin authority only."""
    result = deepcopy(base)
    for key, value in patch.items():
        parts = key.split(".")
        if parts[0] != "coin_overrides":
            continue
        if len(parts) == 1:
            result = deepcopy(value)
            continue
        target = result
        for part in parts[1:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"{key} crosses a non-mapping coin override")
        target[parts[-1]] = deepcopy(value)
    return result


def _selection_sides(patch):
    sides = set()
    for key in patch:
        parts = key.split(".")
        if parts[0] != "coin_overrides":
            continue
        if len(parts) <= 2 or (len(parts) == 3 and parts[2] == "live"):
            sides.update(("long", "short"))
        elif len(parts) >= 4 and parts[2:4] == ["live", "modes"]:
            if len(parts) == 4:
                sides.update(("long", "short"))
            elif parts[4] in {"long", "short"}:
                sides.add(parts[4])
    return sides


def preserve_forced_selection_overrides(document, *, record):
    """Keep generated selections exact when fixed/scenario coin modes replace them.

    A scenario can run over the ordinary root and the optimizer-fixed root.
    Only identifiers normal in every applicable effective context are persisted
    in its shared list. Retained operational normal modes supply context-specific
    selections through the existing backtest flag, without adding false positives.
    """
    root = document.get("coin_overrides", {})
    fixed = document.get("optimize", {}).get("fixed_runtime_overrides", {})
    optimized = _overlay_coin_overrides(root, fixed)

    def emit(patch, contexts, path):
        affected = _selection_sides(patch)
        if not affected:
            return
        normal_sets = [
            forced_normal_coins(_overlay_coin_overrides(base, patch))
            for base in contexts
        ]
        for side in ("long", "short"):
            if side not in affected:
                continue
            first = normal_sets[0][side]
            common = [
                coin
                for coin in first
                if all(coin in values[side] for values in normal_sets[1:])
            ]
            target = f"bot.{side}.forager.always_include"
            if target in patch and patch[target] != common:
                raise ValueError(
                    f"{path}.{target} conflicts with effective released normal selection"
                )
            patch[target] = common
            record(
                f"{path}.{target}",
                None,
                common,
                "preserved effective released normal selection; persist the deterministic context "
                "intersection and retain operational normal flags for context-specific selection",
            )

    emit(fixed, [root], "optimize.fixed_runtime_overrides")
    # Derive optimized coin authority from the original coin controls. Generated
    # bot selection lists above have no effect on the coin-mode map.
    contexts = [root, optimized] if fixed else [root]
    for index, scenario in enumerate(document.get("backtest", {}).get("scenarios", [])):
        emit(
            scenario.get("overrides", {}),
            contexts,
            f"backtest.scenarios[{index}].overrides",
        )
