"""Explicit offline refactoring of directional source policy.

Unlike compact exports these operations deliberately change future inheritance
and may discard held-position policy. Normal loading never invokes them.
"""

from copy import deepcopy
from dataclasses import dataclass

from .authoring import _authored, _patch, _resolved_context
from .inherited_bounds import get_bound_plan, resolve_plan_selectors
from .param_paths import resolve_dotted_config_path
from .scenario_overrides import normalize_scenario_overrides
from .universe import COIN_SOURCE_FIELDS


DISABLED_SIDES = ("preserve", "disable", "omit")
BOT_LAYOUTS = ("preserve", "global", "side")
_SIDES = ("long", "short")
_CAPACITY = {("risk", "max_positions"), ("risk", "total_wallet_exposure_limit_ratio")}
_MISSING = object()


@dataclass
class ReshapeResult:
    config: dict
    changes: list[dict]
    warnings: list[str]


def _prepare(source, base):
    return _resolved_context(source, base_config_path=base, require_complete=False)[0]


def _runnable(config, side):
    policy = config["bot"][side]
    return (policy["enabled"]
            and policy["risk"]["total_wallet_exposure_limit_ratio"] > 0.0
            and round(policy["risk"]["max_positions"]) > 0)


def _transforms(config):
    from optimizer_overrides import optimizer_overrides

    names = config["optimize"].get("enable_overrides", [])
    for side in (None, *_SIDES):
        config = optimizer_overrides(names, config, side)
    return config


def _upper_capacity(config):
    """Inspect capacity reachability using the loader's fixed-selector plan.

    This is not an executable candidate: unrelated cross-field constraints may
    rule out this upper corner. Treating that corner as potentially reachable is
    deliberately conservative, never authority to remove a usable profile.
    """
    result = deepcopy(config)
    plan = get_bound_plan(result)
    values = {binding.key: binding.bound.high for binding in plan.bindings
              if any(target[1] in _SIDES and target[2:] in _CAPACITY
                     for target in binding.effective_targets)}
    # The immutable plan supplies the validated endpoints. This audit is an
    # authored-policy probe, not a candidate replay: local membership resolution
    # can legitimately have refreshed the policy after the plan was prepared.
    return _patch(result, values) if values else result


def _candidate_snapshot(config):
    """Keep completed candidate transforms when scenario ownership begins."""
    from .bot_inheritance import (
        get_bot_policy, install_bot_policy, resolve_bot_policy,
        with_normalized_bot_values, with_prepared_bot_sources,
    )

    result = deepcopy(config)
    policy = get_bot_policy(result)
    snapshot = resolve_bot_policy(policy.candidate_view(), policy.catalog)
    snapshot = with_prepared_bot_sources(snapshot, result["bot"])
    snapshot = with_normalized_bot_values(snapshot, result["bot"])
    install_bot_policy(result, snapshot, effective_bot=result["bot"])
    return result


def _scenario_bound_probe(context, patch):
    """Conservatively include newly declared scenario capacity domains.

    Completed candidate transforms remain authoritative. A newly declared bound
    can broaden the reachability audit, but cannot erase an already positive
    mirrored capacity. Explicit scenario scalar policy retains final precedence.
    """
    normalized = normalize_scenario_overrides(patch)
    if not any(selector in {"optimize", "optimize.bounds"}
               or selector.startswith("optimize.bounds.") for selector in normalized):
        return context
    upper = _upper_capacity(context)
    values = {
        f"bot.{side}.{'.'.join(suffix)}": max(
            context["bot"][side][suffix[0]][suffix[1]],
            upper["bot"][side][suffix[0]][suffix[1]],
        )
        for side in _SIDES for suffix in _CAPACITY
    }
    return _patch(_patch(context, values), patch)


def _contexts(source, base, *, upper=False):
    root = _prepare(source, base)
    yield "root", root
    fixed = root["optimize"].get("fixed_runtime_overrides", {})
    optimized = _patch(root, fixed)
    if upper:
        # A fixed/scenario parent replacement may declare another bounds tree.
        # Probe its current plan, then restore the scalar override precedence.
        optimized = _patch(_upper_capacity(optimized), fixed)
    optimized = _candidate_snapshot(_transforms(optimized))
    if not upper:
        optimized = _prepare(optimized, base)
    yield "optimizer", optimized
    # Stored scenarios are audited even when suite execution is currently off.
    for index, scenario in enumerate(root["backtest"].get("scenarios", [])):
        patch = scenario.get("overrides", {})
        for name, initial in (("root", root), ("optimizer", optimized)):
            context = _patch(initial, patch)
            if name == "optimizer":
                if upper:
                    # A reachability probe may add possibilities, but must never
                    # hide the actual completed-candidate scenario permission.
                    yield f"optimizer scenario {index + 1} before bound probe", context
                    context = _scenario_bound_probe(context, patch)
                # Candidate transforms precede scenario policy. Only coupled
                # spans are refreshed afterwards by the actual suite owner;
                # replaying mirrors/lossless-close would hide scenario values.
                from optimizer_overrides import apply_coupled_unstuck_ema_spans

                context = apply_coupled_unstuck_ema_spans(context)
            if not upper:
                context = _prepare(context, base)
            yield f"{name} scenario {index + 1}", context


def _merge(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _profile(source, prepared, side):
    """Current numerical profile with reloadable universe source bindings."""
    profile = deepcopy(prepared["bot"][side])
    profile.pop("enabled", None)
    bot = source.get("bot", {})
    common = bot.get("global", {}).get("universe", {})
    local = bot.get(side, {}).get("universe", {})
    for field in COIN_SOURCE_FIELDS:
        value = local.get(field, _MISSING)
        if value is _MISSING or field == "approved_coins" and value is None:
            value = common.get(field, [])
        # A null ignore source clears; it does not inherit the new common list.
        if value is None:
            value = []
        profile.setdefault("universe", {})[field] = deepcopy(value)
    return profile


def _omitted_path(path, omitted):
    if any(path[index] == "bot" and path[index + 1] in omitted
           for index in range(len(path) - 1)):
        return True
    return path[:2] == ("optimize", "bounds") and len(path) > 2 and path[2] in omitted


def _prune_value(value, path, omitted, record):
    if _omitted_path(path, omitted):
        record(path, value, None, "omitted dormant directional policy")
        return _MISSING
    if path and path[-1] == "override_config_path" and value and "coin_overrides" in path:
        raise ValueError(
            "cannot omit directional policy while referenced coin override files remain; "
            "export with clean-config --mode resolved first"
        )
    if not isinstance(value, dict):
        return deepcopy(value)
    result = {}
    for key, child in value.items():
        selected = _prune_value(child, (*path, key), omitted, record)
        if selected is not _MISSING:
            result[key] = selected
    return result if result or not value else _MISSING


def _prune_patch(patch, prepared, omitted, record, prefix):
    result = {}
    for selector, value in normalize_scenario_overrides(patch).items():
        path = resolve_dotted_config_path(prepared, selector)
        selected = _prune_value(value, path, omitted,
                                lambda p, old, new, reason: record((*prefix, selector), old, new, reason))
        if selected is not _MISSING:
            result[selector] = selected
    return result


def _omit(source, prepared, omitted, record):
    result = deepcopy(source)
    for side in sorted(omitted):
        old = result["bot"].pop(side, _MISSING)
        if old is not _MISSING:
            record(("bot", side), old, None, "omitted dormant directional profile")
        bounds = result.get("optimize", {}).get("bounds", {})
        old = bounds.pop(side, _MISSING)
        if old is not _MISSING:
            record(("optimize", "bounds", side), old, None, "removed omitted side optimizer bounds")
    patches = result.get("coin_overrides", {})
    for coin, patch in list(patches.items()):
        selected = _prune_value(patch, ("coin_overrides", coin), omitted, record)
        if selected is _MISSING:
            patches.pop(coin)
        else:
            patches[coin] = selected
    optimize = result.get("optimize", {})
    if "fixed_runtime_overrides" in optimize:
        optimize["fixed_runtime_overrides"] = _prune_patch(
            optimize["fixed_runtime_overrides"], prepared, omitted, record,
            ("optimize", "fixed_runtime_overrides"),
        )
    plan = get_bound_plan(prepared)
    bindings = {binding.key: binding for binding in plan.bindings}
    selectors = []
    for selector in optimize.get("fixed_params", []):
        matches = resolve_plan_selectors(prepared, [selector])
        if matches and all(all(target[1] in omitted for target in bindings[key].effective_targets)
                           for key in matches):
            record(("optimize", "fixed_params"), selector, None,
                   "removed fixed selector targeting only omitted sides")
        else:
            selectors.append(selector)
    if "fixed_params" in optimize:
        optimize["fixed_params"] = selectors
    for index, scenario in enumerate(result.get("backtest", {}).get("scenarios", [])):
        if "overrides" in scenario:
            scenario["overrides"] = _prune_patch(
                scenario["overrides"], prepared, omitted, record,
                ("backtest", "scenarios", index, "overrides"),
            )
    return result


def _search_signature(config, retained):
    """Compare variable coordinate fanout, rather than source-mask identity."""
    plan = get_bound_plan(config)
    result = []
    for binding in plan.bindings:
        targets = tuple(target for target in binding.effective_targets
                        if target[1] in retained or target[1] == "portfolio")
        if targets and binding.bound.low != binding.bound.high:
            result.append((targets, binding.bound.low, binding.bound.high, binding.bound.step))
    return sorted(result, key=repr)


def _verify(before, after, retained, base):
    original = dict(_contexts(before, base))
    changed = dict(_contexts(after, base))
    for context, expected in original.items():
        actual = changed[context]
        if expected["bot"]["portfolio"] != actual["bot"]["portfolio"]:
            raise ValueError(f"directional refactoring changes portfolio policy in {context}; preserve bot layout")
        for side in retained:
            if expected["bot"][side] != actual["bot"][side]:
                raise ValueError(
                    f"directional refactoring changes retained {side} policy in {context}; "
                    "use --bot-layout preserve or revise the conflicting overrides"
                )
        if _search_signature(expected, retained) != _search_signature(actual, retained):
            raise ValueError(
                f"directional refactoring changes optimizer coordinate ownership in {context}; "
                "use --bot-layout preserve"
            )
        coins = expected.get("coin_overrides", {})
        other = actual.get("coin_overrides", {})
        for coin in coins.keys() | other.keys():
            for side in retained:
                if coins.get(coin, {}).get("bot", {}).get(side, {}) != other.get(coin, {}).get("bot", {}).get(side, {}):
                    raise ValueError(f"directional refactoring changes retained {side} coin policy for {coin} in {context}")


def reshape_config(source, *, disabled_sides="preserve", bot_layout="preserve", base_config_path=""):
    """Refactor a validated v9 export only when explicitly requested.

    Dormant sides have no entry permission or zero root exposure/position
    capacity. They may be disabled/omitted only when every stored execution
    context and its reachable optimizer capacity also remains dormant.
    """
    if disabled_sides not in DISABLED_SIDES:
        raise ValueError(f"unsupported disabled-sides policy: {disabled_sides}")
    if bot_layout not in BOT_LAYOUTS:
        raise ValueError(f"unsupported bot layout: {bot_layout}")
    if disabled_sides == bot_layout == "preserve":
        return ReshapeResult(deepcopy(source), [], [])
    authored, _ = _authored(source, base_config_path)
    prepared = _prepare(authored, base_config_path)
    changes, warnings = [], []

    def record(path, old, new, reason):
        changes.append(dict(path=".".join(map(str, path)), old=deepcopy(old),
                            new=deepcopy(new), reason=reason))

    dormant = {side for side in _SIDES if not _runnable(prepared, side)}
    if disabled_sides != "preserve":
        for name, context in _contexts(authored, base_config_path, upper=True):
            for side in dormant:
                if _runnable(context, side):
                    raise ValueError(
                        f"cannot {disabled_sides} dormant {side}: {name} can enable positive capacity; "
                        "retain it with --disabled-sides preserve or revise its bounds/overrides"
                    )
        if "long" in dormant and any(
            "mirror_short_from_long" in context["optimize"].get("enable_overrides", [])
            for _, context in _contexts(authored, base_config_path, upper=True)
        ) and disabled_sides == "omit":
            raise ValueError("cannot omit long while mirror_short_from_long consumes its profile; remove that transform first")
    result = deepcopy(authored)
    omitted = dormant if disabled_sides == "omit" else set()
    if omitted:
        result = _omit(result, prepared, omitted, record)
        for side in sorted(omitted):
            if side in authored.get("bot", {}) or changes:
                warnings.append(
                    f"Omitting {side} discards its profile and side-specific search policy; "
                    "held positions use the remaining global/schema policy, and re-enabling does not restore the discarded profile."
                )
    elif disabled_sides == "disable":
        for side in sorted(dormant):
            node = result.setdefault("bot", {}).setdefault(side, {})
            if node.get("enabled") is not False:
                record(("bot", side, "enabled"), node.get("enabled"), False,
                       "explicitly disabled dormant directional policy while retaining its profile")
                node["enabled"] = False
    retained = set(_SIDES) - omitted
    # Verification expects only the declared activation change for disable.
    expected = deepcopy(authored)
    if disabled_sides == "disable":
        for side in dormant:
            expected.setdefault("bot", {}).setdefault(side, {})["enabled"] = False
    if bot_layout != "preserve":
        if result.get("optimize", {}).get("bounds", {}).get("global"):
            raise ValueError("cannot change bot layout with global optimizer bounds; use --bot-layout preserve or declare side bounds first")
        bot = result.setdefault("bot", {})
        before = deepcopy(bot)
        if bot_layout == "global":
            active = [side for side in _SIDES if side in retained and _runnable(prepared, side)]
            if len(active) != 1:
                raise ValueError("--bot-layout global requires exactly one enabled side with positive capacity; preserve layout for two-sided profiles")
            active = active[0]
            bot["global"] = _profile(authored, prepared, active)
            bot[active] = {"enabled": True}
            for side in retained - {active}:
                profile = _profile(authored, prepared, side)
                if "enabled" in bot.get(side, {}):
                    profile["enabled"] = bot[side]["enabled"]
                bot[side] = profile
        else:
            common = bot.pop("global", {})
            for side in _SIDES:
                if side in omitted:
                    continue
                # Side-owned layout deliberately pins common values, including
                # dormant held-position settings, without expanding fallbacks.
                bot[side] = _merge(deepcopy(common), bot.get(side, {}))
                universe = bot[side].get("universe", {})
                if universe.get("approved_coins", _MISSING) is None:
                    universe["approved_coins"] = deepcopy(common.get("universe", {}).get("approved_coins", []))
        if bot != before:
            record(("bot",), before, bot, f"refactored directional defaults to {bot_layout} layout")
            warnings.append(f"The {bot_layout} bot layout changes future inheritance; review global edits and side activation before reuse.")
    _verify(expected, result, retained, base_config_path)
    return ReshapeResult(result, changes, warnings)
