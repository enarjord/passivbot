"""Configuration boundary for the opt-in revised HSL engine.

Missing restart choices remain explicit nulls while a scope is disabled, so later
CLI/scenario enablement cannot mistake hydration for an operator's policy choice.
"""
from copy import deepcopy
import logging
import math
import re

from .shared_bot import canonicalize_shared_bot_side

ENGINES = frozenset({"legacy", "revised"})
REMOVED_FIELDS = frozenset({"tier_ratios", "orange_tier_mode", "no_restart_drawdown_threshold"})
FIELDS = frozenset({"enabled", "red_threshold", "ema_span_minutes", "panic_close_order_type",
                    "cooldown_minutes_after_red", "restart_after_red_policy"})


def engine(config):
    value = config.get("live", {}).get("hsl_engine", "legacy")
    if not isinstance(value, str) or value.strip().lower() not in ENGINES:
        raise ValueError("live.hsl_engine must be legacy or revised")
    return value.strip().lower()


def _mode(config):
    value = config.get("live", {}).get("hsl_signal_mode", "coin")
    if not isinstance(value, str) or value.strip().lower() not in {"coin", "pside", "unified"}:
        raise ValueError("live.hsl_signal_mode must be coin, pside or unified")
    return value.strip().lower()


def normalization_template(template, config):
    """Engine-specific hydration defaults, never a new-config authorization."""
    if engine(config) == "legacy":
        return template
    result = deepcopy(template)
    result["live"]["hsl_engine"] = "revised"
    for side in ("long", "short"):
        block = result["bot"][side]["hsl"]
        for key in REMOVED_FIELDS:
            block.pop(key, None)
        block["restart_after_red_policy"] = None
        if _mode(config) == "unified":
            result["optimize"]["bounds"].get(side, {}).pop("hsl", None)
    # Inherited optimizer policy defaults cannot satisfy explicit migration.
    overrides = result["optimize"]["fixed_runtime_overrides"]
    for key in list(overrides):
        if _is_hsl_path(key):
            del overrides[key]
    return result


def generated_template(template, mode="coin"):
    """Explicitly generate a new revised config with authored restart choices."""
    config = deepcopy(template)
    config["live"].update(hsl_engine="revised", hsl_signal_mode=mode)
    config = normalization_template(config, config)
    for side in ("long", "short"):
        config["bot"][side]["hsl"]["restart_after_red_policy"] = "always"
    if mode == "unified":
        config["bot"]["hsl"] = deepcopy(config["bot"]["long"]["hsl"])
        config["optimize"]["bounds"]["hsl"] = deepcopy(template["optimize"]["bounds"]["long"]["hsl"])
        config["optimize"]["fixed_runtime_overrides"]["bot.hsl.restart_after_red_policy"] = "always"
    else:
        for side in ("long", "short"):
            config["optimize"]["fixed_runtime_overrides"][f"bot.{side}.hsl.restart_after_red_policy"] = "always"
    return config


def _number(value, path, *, minimum, maximum=None, strict=False):
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if (not math.isfinite(number) or number < minimum or (strict and number == minimum)
            or (maximum is not None and number > maximum)):
        raise ValueError(f"{path} is outside its finite supported range")
    return number


def normalize_block(block, defaults, path, *, active=True, portfolio=False, verbose=True):
    if not isinstance(block, dict):
        raise TypeError(f"{path} must be a mapping")
    if portfolio:
        required = FIELDS - {"restart_after_red_policy"}
        missing = required - block.keys()
        if missing:
            raise ValueError(f"{path} missing required fields: {', '.join(sorted(missing))}")
    for key in REMOVED_FIELDS:
        if key in block:
            if verbose:
                logging.warning("%s.%s is removed in revised HSL and has no trading effect", path, key)
            del block[key]
    unknown = block.keys() - FIELDS
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    # All ordinary defaults are canonical; restart needs an explicit choice.
    for key in FIELDS - {"restart_after_red_policy"}:
        if key not in block:
            block[key] = deepcopy(defaults[key])
    block.setdefault("restart_after_red_policy", None)
    if type(block["enabled"]) is not bool:
        raise TypeError(f"{path}.enabled must be a boolean")
    policy = block["restart_after_red_policy"]
    if isinstance(policy, str):
        policy = policy.strip().lower()
    if policy is not None and not isinstance(policy, str):
        raise ValueError(f"{path}.restart_after_red_policy must be always or never")
    if active and block["enabled"] and policy not in {"always", "never"}:
        raise ValueError(f"{path}.restart_after_red_policy requires an explicit choice of always or never; threshold is removed")
    if policy not in {None, "always", "never", "threshold"}:
        raise ValueError(f"{path}.restart_after_red_policy must be always or never")
    block["restart_after_red_policy"] = policy
    block["ema_span_minutes"] = _number(block["ema_span_minutes"], f"{path}.ema_span_minutes", minimum=1)
    block["red_threshold"] = _number(block["red_threshold"], f"{path}.red_threshold", minimum=0, maximum=1, strict=True)
    block["cooldown_minutes_after_red"] = _number(block["cooldown_minutes_after_red"], f"{path}.cooldown_minutes_after_red", minimum=0)
    order_type = block["panic_close_order_type"]
    if not isinstance(order_type, str) or order_type.strip().lower() not in {"limit", "market"}:
        raise ValueError(f"{path}.panic_close_order_type must be limit or market")
    block["panic_close_order_type"] = order_type.strip().lower()


def _is_hsl_path(path):
    return bool(re.search(r"(^|[._])hsl([._]|$)", path))


def validate_parameter_path(path, mode):
    if not _is_hsl_path(path):
        return
    if any(key in path for key in REMOVED_FIELDS):
        raise ValueError(f"{path} targets a removed revised HSL parameter; remove this bound/override")
    if mode == "unified" and re.search(r"(^|[._])(long|short)[._]hsl([._]|$)", path):
        raise ValueError(f"{path} is inactive in unified mode; use explicit bot.hsl portfolio parameter paths")


def _walk_paths(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_paths(value, (*prefix, str(key)))
    else:
        yield ".".join(prefix), node


def validate_revised_paths(config):
    mode = _mode(config)
    optimize = config.get("optimize", {})
    for section in ("bounds", "fixed_runtime_overrides"):
        for path, _ in _walk_paths(optimize.get(section, {})):
            validate_parameter_path(path, mode)
    def strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from strings(value)
    for metric in strings({k: optimize.get(k, {}) for k in ("scoring", "limits")}):
        if "hard_stop_time_in_yellow" in metric or "hard_stop_time_in_orange" in metric:
            raise ValueError(f"{metric} is removed for revised HSL; choose a supported objective/limit")


def validate_override_paths(config, overrides, *, allow_engine=False):
    """Check explicit patch paths using the final selected engine and scope mode."""
    leaves = dict(_walk_paths(overrides))
    if "live.hsl_engine" in leaves and not allow_engine:
        raise ValueError("live.hsl_engine is a startup selector, not a scenario or optimizer override")
    selected = {"live": dict(config.get("live", {}))}
    for key in ("hsl_engine", "hsl_signal_mode"):
        if f"live.{key}" in leaves:
            selected["live"][key] = leaves[f"live.{key}"]
    if engine(selected) != "revised":
        return
    for path in leaves:
        validate_parameter_path(path, _mode(selected))
    validate_revised_paths({**selected, "optimize": {"scoring": overrides}})


def normalize_revised(config, template, *, verbose=True):
    """Run before hydration, and again for effective CLI/scenario configurations."""
    selected = engine(config)
    config.setdefault("live", {})["hsl_engine"] = selected
    if selected == "legacy":
        if "hsl" in config.get("bot", {}):
            raise ValueError("bot.hsl is a revised unified block; select revised or supply a legacy-compatible config")
        return
    mode = _mode(config)
    config["live"]["hsl_signal_mode"] = mode
    bot = config.setdefault("bot", {})
    if mode == "unified" and "hsl" not in bot:
        raise ValueError("revised unified HSL requires an explicit bot.hsl block; supply it or select pside for separate side controllers")
    validate_revised_paths(config)
    enabled = False
    for side in ("long", "short"):
        side_cfg = bot.setdefault(side, {})
        canonicalize_shared_bot_side(side_cfg, path_prefix=("bot", side), seed_missing_groups=True)
        normalize_block(side_cfg["hsl"], template["bot"][side]["hsl"], f"bot.{side}.hsl",
                        active=mode != "unified", verbose=verbose)
        enabled |= mode != "unified" and side_cfg["hsl"]["enabled"]
    if "hsl" in bot:
        normalize_block(bot["hsl"], {}, "bot.hsl", active=mode == "unified", portfolio=True, verbose=verbose)
        enabled |= mode == "unified" and bot["hsl"]["enabled"]
    for coin, patch in config.get("coin_overrides", {}).items():
        for side, values in patch.get("bot", {}).items():
            if side not in {"long", "short"} or not isinstance(values, dict):
                continue
            canonicalize_shared_bot_side(values)
            if mode != "coin" and values.get("hsl"):
                raise ValueError(f"coin_overrides.{coin}.bot.{side}.hsl is inactive outside coin mode")
            for path, _ in _walk_paths(values.get("hsl", {})):
                validate_parameter_path(f"coin_overrides.{coin}.bot.{side}.hsl.{path}", mode)
            if mode == "coin" and values.get("hsl"):
                effective = {**bot[side]["hsl"], **values["hsl"]}
                normalize_block(effective, {}, f"coin_overrides.{coin}.bot.{side}.hsl", verbose=verbose)
                enabled |= effective["enabled"]
    intervention = config["live"].get("hsl_position_during_cooldown_policy", "panic")
    if not isinstance(intervention, str) or intervention.strip().lower() not in {"panic", "normal"}:
        raise ValueError("revised live.hsl_position_during_cooldown_policy requires panic or normal; manual/tp_only/graceful_stop are removed")
    config["live"]["hsl_position_during_cooldown_policy"] = intervention.strip().lower()
    if enabled:
        value = config["live"].get("pnls_max_lookback_days", template["live"]["pnls_max_lookback_days"])
        config["live"]["pnls_max_lookback_days"] = _number(value, "enabled revised HSL lookback days [1,90]", minimum=1, maximum=90)


def require_runtime_support(config, supported_modes=()):
    if engine(config) == "revised" and _mode(config) not in supported_modes:
        raise ValueError(f"revised HSL {_mode(config)} runtime integration is not available in this build; legacy remains the default")


def _side_has_enabled_policy(config, side, markets_by_exchange):
    base_enabled = config["bot"][side]["hsl"]["enabled"]
    if _mode(config) != "coin":
        return base_enabled
    datasets = config.get("backtest", {}).get("coins")
    if datasets is None:
        # Config-only callers can check the declared policies before a dataset
        # exists. Runtime callers below use only their actual dataset members.
        return base_enabled or any(
            patch.get("bot", {}).get(side, {}).get("hsl", {}).get("enabled", base_enabled)
            for patch in config.get("coin_overrides", {}).values()
        )
    from backtest import _get_backtest_coin_override

    for exchange, coins in datasets.items():
        markets = (markets_by_exchange or {}).get(exchange, {})
        for coin in coins:
            patch = _get_backtest_coin_override(config, markets, exchange, coin)
            if patch.get("bot", {}).get(side, {}).get("hsl", {}).get("enabled", base_enabled):
                return True
    return False


def validate_optimizer_metrics(config, metrics, *, markets_by_exchange=None):
    """Validate only objectives/limits consuming this effective scenario's results."""
    if engine(config) != "revised":
        return
    from .metrics import canonical_metric_name, canonicalize_metric_name, split_metric_stat_suffix

    for metric in metrics:
        name = canonicalize_metric_name(metric)
        name = name.removesuffix("_usd").removesuffix("_btc")
        name, _ = split_metric_stat_suffix(canonical_metric_name(name))
        if name.startswith(("hard_stop_time_in_yellow", "hard_stop_time_in_orange")):
            raise ValueError(f"{metric} is removed for revised HSL; choose a supported objective/limit")
        side_signal = name.endswith(("_long", "_short")) and (
            name.startswith("hard_stop_")
            or name.startswith("drawdown_worst_ema_strategy_eq_")
            or name.startswith("drawdown_worst_mean_1pct_ema_strategy_eq_")
        )
        if _mode(config) == "unified" and side_signal:
            raise ValueError(f"{metric} has no side controller in revised unified HSL; use the portfolio metric")
        if side_signal and not _side_has_enabled_policy(config, name.rsplit("_", 1)[1], markets_by_exchange):
            raise ValueError(f"{metric} has no enabled side controller in this revised HSL scenario")
