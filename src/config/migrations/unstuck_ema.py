"""Materialize legacy strategy-coupled unstuck spans before defaults are applied."""

import logging
from copy import deepcopy

from optimization.bounds import Bound

from ..strategy_spec import (
    get_strategy_defaults,
    get_strategy_optimize_bounds,
    normalize_strategy_kind,
)
from ..shared_bot import canonicalize_shared_bot_side


def _migrate_inactive_zero_span(value, bot, default, path):
    unstuck = bot.get("unstuck", {})
    if value == 0 and (
        bot.get("risk", {}).get("total_wallet_exposure_limit") == 0
        or bot.get("wallet_exposure_limit") == 0
        or unstuck.get("enabled") is False
        or unstuck.get("ema_gating_enabled") is False
    ):
        logging.warning(
            "[config] Cannot copy zero strategy span to %s; "
            "using the positive strategy default %s for this inactive side/gate. "
            "Current inactive behavior is unchanged; review spans before enabling it.",
            path,
            default,
        )
        return default
    return value


def migrate_unstuck_ema_spans(
    config: dict,
    *,
    base_config_path: str = "",
    verbose: bool = True,
    tracker=None,
    explicit_bounds=None,
) -> None:
    kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    defaults = get_strategy_defaults(kind)
    migrated = {}
    for side, bot in config.get("bot", {}).items():
        if side not in ("long", "short") or not isinstance(bot, dict):
            continue
        canonicalize_shared_bot_side(bot)
        unstuck = bot.setdefault("unstuck", {})
        strategy = bot.get("strategy", {}).get(kind, {})
        for key in ("ema_span_0", "ema_span_1"):
            if key in unstuck:
                continue
            # Missing strategy leaves have always been populated from strategy defaults.
            value = deepcopy(strategy.get(key, defaults[side][key]))
            value = _migrate_inactive_zero_span(
                value, bot, defaults[side][key], f"bot.{side}.unstuck.{key}"
            )
            unstuck[key] = value
            migrated.setdefault(side, []).append(key)
            if tracker is not None:
                tracker.add(["bot", side, "unstuck", key], value)
    if not migrated:
        return

    # Resolve file + inline strategy precedence before pinning only overridden spans.
    # Persist the result inline so reloading the migrated config is idempotent.
    from ..overrides import (
        load_override_config,
        _unwrap_override_document,
        nested_update,
        parse_old_coin_flags,
    )

    if not config.get("coin_overrides") and config.get("live", {}).get("coin_flags"):
        config["coin_overrides"] = parse_old_coin_flags(config)
        if tracker is not None and config["coin_overrides"]:
            tracker.add(["coin_overrides"], deepcopy(config["coin_overrides"]))

    pinned = 0
    for coin, override in config.get("coin_overrides", {}).items():
        if not isinstance(override, dict):
            continue  # Normal override validation reports the malformed object.
        effective_patch = {}
        if override.get("override_config_path"):
            loader_config = deepcopy(config)
            loader_config.setdefault("live", {})["base_config_path"] = base_config_path
            source = load_override_config(loader_config, coin)
            effective_patch = deepcopy(
                _unwrap_override_document(source, source=f"coin_overrides.{coin}")
            )
        nested_update(effective_patch, deepcopy(override))
        for side, keys in migrated.items():
            patch_side = effective_patch.get("bot", {}).get(side, {})
            canonicalize_shared_bot_side(patch_side)
            patch_unstuck = patch_side.get("unstuck", {})
            strategy = patch_side.get("strategy", {}).get(kind, {})
            for key in keys:
                if key in patch_unstuck or key not in strategy:
                    continue
                target = (
                    override.setdefault("bot", {})
                    .setdefault(side, {})
                    .setdefault("unstuck", {})
                )
                effective_side = deepcopy(config["bot"][side])
                nested_update(effective_side, deepcopy(patch_side))
                target[key] = _migrate_inactive_zero_span(
                    deepcopy(strategy[key]),
                    effective_side,
                    defaults[side][key],
                    f"coin_overrides.{coin}.bot.{side}.unstuck.{key}",
                )
                pinned += 1
                if tracker is not None:
                    tracker.add(
                        ["coin_overrides", coin, "bot", side, "unstuck", key],
                        target[key],
                    )

    # The old search had one gene driving two consumers. No pair of independent
    # ranges can encode that equality constraint. Freeze new genes at the migrated
    # starting values for varying ranges, or the old fixed value; explicit new bounds win.
    bounds = config.get("optimize", {}).get("bounds")
    coupled_search = []
    if isinstance(bounds, dict):
        default_strategy_bounds = get_strategy_optimize_bounds(kind)
        for side, keys in migrated.items():
            side_bounds = bounds.setdefault(side, {})
            unstuck_bounds = side_bounds.setdefault("unstuck", {})
            strategy_bounds = side_bounds.get("strategy", {}).get(kind, {})
            for key in keys:
                old_bound = Bound.from_config(
                    f"{side}_{key}",
                    strategy_bounds.get(key, default_strategy_bounds[side][key]),
                )
                supplied_bounds = bounds if explicit_bounds is None else explicit_bounds
                supplied_unstuck = supplied_bounds.get(side, {}).get("unstuck", {})
                if key not in supplied_unstuck:
                    value = config["bot"][side]["unstuck"][key]
                    if old_bound.low == old_bound.high:
                        value = _migrate_inactive_zero_span(
                            old_bound.low,
                            config["bot"][side],
                            value,
                            f"optimize.bounds.{side}.unstuck.{key}",
                        )
                    unstuck_bounds[key] = [value, value]
                    if tracker is not None:
                        tracker.add(
                            ["optimize", "bounds", side, "unstuck", key], [value, value]
                        )
                if old_bound.low != old_bound.high:
                    coupled_search.append(f"{side}.{key}")
    if coupled_search:
        logging.warning(
            "[config] Unstuck EMA migration preserves this config's trading spans, but a 1-to-1 "
            "optimizer-search migration is impossible: strategy spans previously also controlled "
            "unstucking (%s). New unstuck span bounds are fixed at migrated values unless explicitly "
            "provided. Set optimize.bounds.<side>.unstuck.ema_span_0/1 to tune them independently; "
            "previous optimizer checkpoints must start a new search.",
            ", ".join(coupled_search),
        )
    if pinned:
        logging.info(
            "[config] Preserved %d coin-specific unstuck EMA span overrides. To optimize a shared "
            "portfolio pair, deliberately remove coin_overrides.<coin>.bot.<side>.unstuck.ema_span_0/1 "
            "and set the global unstuck bounds.",
            pinned,
        )
    if verbose:
        logging.info(
            "[config] Migrated unstuck EMA spans from each side's effective %s strategy; saved trading behavior is preserved.",
            kind,
        )

    def scenario_leaves(mapping, prefix=""):
        for key, value in mapping.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from scenario_leaves(value, path)
            else:
                yield path, value

    # Nested and dotted scenario overrides have the same exact leaf migration.
    # More general strategy replacement cannot be represented by copying a leaf.
    for scenario in config.get("backtest", {}).get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        overrides = scenario.get("overrides", {})
        if not isinstance(overrides, dict):
            continue
        leaves = dict(scenario_leaves(overrides))
        for side, keys in migrated.items():
            for key in keys:
                for old_path in (
                    f"bot.{side}.strategy.{kind}.{key}",
                    f"bot.{side}.{key}",
                ):
                    new_path = f"bot.{side}.unstuck.{key}"
                    if old_path in leaves and new_path not in leaves:
                        overrides[new_path] = deepcopy(leaves[old_path])
        if "live.strategy_kind" in leaves or any(
            isinstance(v, dict) and "strategy" in str(k) for k, v in overrides.items()
        ):
            logging.warning(
                "[config] Unstuck EMA migration cannot guarantee 1-to-1 behavior for scenario %r "
                "which replaces strategy settings. Set explicit bot.<side>.unstuck.ema_span_0/1 "
                "scenario overrides to match its intended horizons.",
                scenario.get("label", "<unnamed>"),
            )
