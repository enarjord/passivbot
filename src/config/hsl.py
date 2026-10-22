"""Configuration boundary for the v9 decaying-risk HSL engine."""

from copy import deepcopy
import logging
import math
import re

from .shared_bot import canonicalize_shared_bot_side

REMOVED_FIELDS = frozenset(
    {"tier_ratios", "orange_tier_mode", "no_restart_drawdown_threshold"}
)
REMOVED_LIVE_FIELDS = frozenset(
    {
        "hsl_accept_incomplete_history",
        "hsl_unavailable_grace_seconds",
        "risk_input_max_attempts",
    }
)
FIELDS = frozenset(
    {
        "enabled",
        "scale_budget_with_excess_allowance",
        "drawdown_ratio_threshold",
        "drawdown_ratio_ema_span_minutes",
        "memory_ema_span_minutes",
        "panic_close_order_type",
        "cooldown_after_red_minutes",
    }
)


def _authored_paths(node, path=""):
    """Yield raw leaves, including scenario lists and dotted override paths."""
    if isinstance(node, dict):
        for key, value in node.items():
            if not str(key).startswith("_"):
                yield from _authored_paths(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for value in node:
            yield from _authored_paths(value, path)
    else:
        yield path, node


def _requires_hsl_migration(node, path=""):
    """Inspect authored policies before hydration can erase their schema origin."""
    for path, value in _authored_paths(node, path):
        if not _is_hsl_path(path):
            continue
        if path.endswith("enabled") and value not in (False, None, 0):
            return True
    return False


def require_current_hsl_schema(config, *, base_config_path="", additional_inputs=()):
    """Reject v8 policy before hydration or CLI/scenario enablement can erase it."""
    from .migrations.v9 import require_v9_runtime_config

    require_v9_runtime_config(config, additional_inputs=additional_inputs)
    from .overrides import load_override_config, _validated_file_patch, _resolve_override_config_path
    from .parse import load_raw_config

    for node in (config, *additional_inputs):
        for path, value in _authored_paths(node):
            if "coin_overrides" in path and path.endswith("override_config_path"):
                reference = value
            else:
                continue
            context = {
                "coin_overrides": {"input": {"override_config_path": reference}},
                "live": {"base_config_path": base_config_path or config.get("live", {}).get("base_config_path", "")},
            }
            try:
                authored = load_override_config(context, "input", config_loader=load_raw_config)
                require_v9_runtime_config(config, additional_inputs=(authored,))
                _validated_file_patch(
                    authored, config=config, coin="input", source=path,
                    resolved_path=_resolve_override_config_path(context, "input"),
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"{path} not found: {reference!r}") from exc
            except OSError as exc:
                raise type(exc)(f"{path} {reference!r} could not be read: {exc}") from exc
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path} {reference!r} is invalid: {exc}") from exc


def clear_legacy_hsl_authorization(node, path=""):
    """Clear dormant legacy authority in a copy, including dotted patches."""
    if isinstance(node, dict):
        for key in list(node):
            if str(key).startswith("_"):
                continue
            leaf_path = f"{path}.{key}" if path else str(key)
            if leaf_path.endswith("live.hsl_engine"):
                del node[key]
            elif _is_hsl_path(leaf_path) and leaf_path.endswith("restart_after_red_policy"):
                del node[key]
            else:
                clear_legacy_hsl_authorization(node[key], leaf_path)
    elif isinstance(node, list):
        for value in node:
            clear_legacy_hsl_authorization(value, path)


def engine(config):
    if "hsl_engine" not in config.get("live", {}):
        return "hsl"
    value = config["live"]["hsl_engine"]
    if isinstance(value, str) and value.strip().lower() == "legacy":
        raise ValueError(
            "legacy HSL has been removed; migrate this config with passivbot tool migrate-config-v9 and re-backtest before use"
        )
    raise ValueError(
        "live.hsl_engine is obsolete; migrate this config with passivbot tool migrate-config-v9 and re-backtest before use"
    )


def _mode(config):
    value = config.get("bot", {}).get("portfolio", {}).get("hsl", {}).get("scope", "coin")
    if not isinstance(value, str) or value.strip().lower() not in {
        "coin",
        "position_side",
        "portfolio",
    }:
        raise ValueError("bot.portfolio.hsl.scope must be coin, position_side or portfolio")
    return value.strip().lower()


def normalization_template(template, config):
    """HSL hydration defaults, never a new-config authorization."""
    engine(config)
    result = deepcopy(template)
    result["live"].pop("hsl_engine", None)
    result["live"].pop("hsl_position_during_cooldown_policy", None)
    for side in ("long", "short"):
        block = result["bot"][side]["hsl"]
        for key in REMOVED_FIELDS:
            block.pop(key, None)
        if _mode(config) == "portfolio":
            result["optimize"]["bounds"].get(side, {}).pop("hsl", None)
    # Inherited optimizer policy defaults cannot satisfy explicit migration.
    overrides = result["optimize"]["fixed_runtime_overrides"]
    for key in list(overrides):
        if _is_hsl_path(key):
            del overrides[key]
    return result


def generated_template(template, mode="coin"):
    """Generate an explicit v9 HSL configuration."""
    config = deepcopy(template)
    config["bot"]["portfolio"]["hsl"]["scope"] = mode
    config = normalization_template(config, config)
    if mode == "portfolio":
        config["bot"]["portfolio"]["hsl"].update(deepcopy(config["bot"]["long"]["hsl"]))
        config["optimize"]["bounds"].setdefault("portfolio", {})["hsl"] = deepcopy(
            template["optimize"]["bounds"]["long"]["hsl"]
        )
    return config


def _number(value, path, *, minimum, maximum=None, strict=False):
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if (
        not math.isfinite(number)
        or number < minimum
        or (strict and number == minimum)
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(f"{path} is outside its finite supported range")
    return number


def normalize_block(
    block, defaults, path, *, active=True, portfolio=False, verbose=True
):
    if not isinstance(block, dict):
        raise TypeError(f"{path} must be a mapping")
    if portfolio:
        required = FIELDS - {"scale_budget_with_excess_allowance"}
        missing = required - block.keys()
        if missing:
            raise ValueError(
                f"{path} missing required fields: {', '.join(sorted(missing))}"
            )
    for key in REMOVED_FIELDS:
        if key in block:
            if verbose:
                logging.warning(
                    "%s.%s is removed in HSL and has no trading effect", path, key
                )
            del block[key]
    unknown = block.keys() - FIELDS - ({"scope"} if portfolio else set())
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    # All ordinary defaults are canonical.
    for key in FIELDS:
        if key not in block:
            block[key] = (
                False if key == "scale_budget_with_excess_allowance"
                else deepcopy(defaults[key])
            )
    if type(block["enabled"]) is not bool:
        raise TypeError(f"{path}.enabled must be a boolean")
    if type(block["scale_budget_with_excess_allowance"]) is not bool:
        raise TypeError(f"{path}.scale_budget_with_excess_allowance must be a boolean")
    from .risk_memory import normalize_memory_ema_span_minutes

    for field in ("drawdown_ratio_ema_span_minutes", "memory_ema_span_minutes"):
        block[field] = normalize_memory_ema_span_minutes(
            block[field], field=f"{path}.{field}"
        )
    block["drawdown_ratio_threshold"] = _number(
        block["drawdown_ratio_threshold"],
        f"{path}.drawdown_ratio_threshold",
        minimum=0,
        maximum=1,
        strict=True,
    )
    block["cooldown_after_red_minutes"] = _number(
        block["cooldown_after_red_minutes"],
        f"{path}.cooldown_after_red_minutes",
        minimum=0,
    )
    order_type = block["panic_close_order_type"]
    if not isinstance(order_type, str) or order_type.strip().lower() not in {
        "limit",
        "market",
    }:
        raise ValueError(f"{path}.panic_close_order_type must be limit or market")
    block["panic_close_order_type"] = order_type.strip().lower()


def _is_hsl_path(path):
    return bool(re.search(r"(^|[._])hsl([._]|$)", path))


def validate_parameter_path(path, mode):
    if "hsl_position_during_cooldown_policy" in path:
        raise ValueError(f"{path} is removed in HSL; exposure clears cooldown")
    if any(key in path for key in REMOVED_LIVE_FIELDS):
        raise ValueError(
            f"{path} is a removed HSL recovery control; remove this bound/override"
        )
    if ("restart_after_red_policy" in path or "pnls_max_lookback_days" in path
            or "realized_loss_gate_lookback_minutes" in path):
        raise ValueError(f"{path} is removed in v9; use passivbot tool migrate-config-v9")
    if not _is_hsl_path(path):
        return
    if any(key in path for key in REMOVED_FIELDS):
        raise ValueError(
            f"{path} targets a removed HSL parameter; remove this bound/override"
        )
    if mode == "portfolio" and re.search(r"(^|[._])(long|short)[._]hsl([._]|$)", path):
        raise ValueError(
            f"{path} is inactive in portfolio mode; use explicit bot.portfolio.hsl portfolio parameter paths"
        )


def _walk_paths(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_paths(value, (*prefix, str(key)))
    else:
        yield ".".join(prefix), node


def validate_hsl_paths(config):
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
            raise ValueError(
                f"{metric} is removed for HSL; choose a supported objective/limit"
            )


def validate_override_paths(config, overrides, *, allow_engine=False):
    """Check explicit patch paths using the final selected engine and scope mode."""
    leaves = dict(_walk_paths(overrides))
    if "live.hsl_engine" in leaves and not allow_engine:
        raise ValueError(
            "live.hsl_engine is a startup selector, not a scenario or optimizer override"
        )
    selected = deepcopy(config)
    if "bot.portfolio.hsl.scope" in leaves:
        selected.setdefault("bot", {}).setdefault("portfolio", {}).setdefault("hsl", {})["scope"] = leaves["bot.portfolio.hsl.scope"]
    for key in ("hsl_engine",):
        if f"live.{key}" in leaves:
            selected["live"][key] = leaves[f"live.{key}"]
    # CLI mutation precedes the schema gate: an explicit disable must be able
    # to retire a legacy policy before loading. Other patches remain strict.
    if not allow_engine:
        engine(selected)
    for path in leaves:
        validate_parameter_path(path, _mode(selected))
    validate_hsl_paths({**selected, "optimize": {"scoring": overrides}})


def normalize_hsl(config, template, *, verbose=True):
    """Run before hydration, and again for effective CLI/scenario configurations."""
    engine(config)
    config.setdefault("live", {}).pop("hsl_engine", None)
    from .risk_memory import normalize_risk_memory_config

    config["live"].setdefault("startup", {}).setdefault(
        "risk_bootstrap_max_lookback_minutes",
        deepcopy(template["live"]["startup"]["risk_bootstrap_max_lookback_minutes"]),
    )
    bot = config.setdefault("bot", {})
    bot.setdefault("portfolio", deepcopy(template["bot"]["portfolio"]))
    bot["portfolio"].setdefault("hsl", {"scope": "coin"})
    bot["portfolio"].setdefault("loss_gate", deepcopy(template["bot"]["portfolio"]["loss_gate"]))
    bot["portfolio"].setdefault("unstuck", deepcopy(template["bot"]["portfolio"]["unstuck"]))
    for name, block in (("portfolio.unstuck", bot["portfolio"]["unstuck"]), ("portfolio.loss_gate", bot["portfolio"]["loss_gate"])):
        defaults = template["bot"]["portfolio"]["unstuck"] if name == "portfolio.unstuck" else template["bot"]["portfolio"]["loss_gate"]
        for key, value in defaults.items():
            block.setdefault(key, deepcopy(value))
    normalize_risk_memory_config(config)
    mode = _mode(config)
    bot["portfolio"]["hsl"]["scope"] = mode
    if mode != "portfolio" and set(bot["portfolio"]["hsl"]) - {"scope"}:
        raise ValueError("bot.portfolio.hsl policy fields require portfolio scope; use bot.<side>.hsl in coin/position_side scope")
    validate_hsl_paths(config)
    enabled = False
    for side in ("long", "short"):
        side_cfg = bot.setdefault(side, {})
        for key in REMOVED_FIELDS:
            if f"hsl_{key}" in side_cfg:
                if verbose:
                    logging.warning(
                        "bot.%s.hsl_%s is removed and has no trading effect", side, key
                    )
                del side_cfg[f"hsl_{key}"]
        canonicalize_shared_bot_side(
            side_cfg, path_prefix=("bot", side), seed_missing_groups=True
        )
        normalize_block(
            side_cfg["hsl"],
            template["bot"][side]["hsl"],
            f"bot.{side}.hsl",
            active=mode != "portfolio",
            verbose=verbose,
        )
        if mode != "coin" and side_cfg["hsl"]["scale_budget_with_excess_allowance"]:
            raise ValueError(f"bot.{side}.hsl.scale_budget_with_excess_allowance requires coin HSL mode")
        enabled |= mode != "portfolio" and side_cfg["hsl"]["enabled"]
    if mode == "portfolio":
        normalize_block(
            bot["portfolio"]["hsl"],
            {},
            "bot.portfolio.hsl",
            active=mode == "portfolio",
            portfolio=True,
            verbose=verbose,
        )
        if bot["portfolio"]["hsl"]["scale_budget_with_excess_allowance"]:
            raise ValueError("bot.portfolio.hsl.scale_budget_with_excess_allowance requires coin HSL mode")
        enabled |= mode == "portfolio" and bot["portfolio"]["hsl"]["enabled"]
    for coin, patch in config.get("coin_overrides", {}).items():
        for side, values in patch.get("bot", {}).items():
            if side not in {"long", "short"} or not isinstance(values, dict):
                continue
            canonicalize_shared_bot_side(values)
            if mode != "coin" and values.get("hsl"):
                raise ValueError(
                    f"coin_overrides.{coin}.bot.{side}.hsl is inactive outside coin mode"
                )
            for path, _ in _walk_paths(values.get("hsl", {})):
                validate_parameter_path(
                    f"coin_overrides.{coin}.bot.{side}.hsl.{path}", mode
                )
            if mode == "coin" and values.get("hsl"):
                effective = {**bot[side]["hsl"], **values["hsl"]}
                normalize_block(
                    effective,
                    {},
                    f"coin_overrides.{coin}.bot.{side}.hsl",
                    verbose=verbose,
                )
                enabled |= effective["enabled"]
    for key in sorted(REMOVED_LIVE_FIELDS):
        if key in config["live"]:
            if verbose:
                logging.warning(
                    "live.%s is removed; HSL evaluates best-effort history without an emergency fallback",
                    key,
                )
            del config["live"][key]
    if "hsl_position_during_cooldown_policy" in config["live"]:
        if verbose:
            logging.warning(
                "live.hsl_position_during_cooldown_policy is removed in HSL; exposure clears cooldown"
            )
        del config["live"]["hsl_position_during_cooldown_policy"]


def require_live_runtime_support(config):
    """Validate live replay capability and return the required cooldown horizon."""
    from .risk_memory import normalize_risk_bootstrap_max_lookback_minutes

    cap = normalize_risk_bootstrap_max_lookback_minutes(
        config["live"]["startup"]["risk_bootstrap_max_lookback_minutes"]
    )
    mode = _mode(config)
    policies = []
    if mode == "portfolio":
        policies.append(("bot.portfolio.hsl", config["bot"]["portfolio"]["hsl"]))
    else:
        for side in ("long", "short"):
            base = config["bot"][side]["hsl"]
            policies.append((f"bot.{side}.hsl", base))
            if mode == "coin":
                for coin, patch in config.get("coin_overrides", {}).items():
                    hsl = patch.get("bot", {}).get(side, {}).get("hsl", {})
                    if hsl:
                        policies.append((f"coin_overrides.{coin}.bot.{side}.hsl", {**base, **hsl}))
    required = 0.0
    for path, policy in policies:
        if not policy["enabled"]:
            continue
        required = max(required, policy["cooldown_after_red_minutes"])
        if policy["cooldown_after_red_minutes"] > cap:
            raise ValueError(
                f"{path}.cooldown_after_red_minutes exceeds live.startup.risk_bootstrap_max_lookback_minutes; "
                "live cannot reconstruct its full finite cooldown within this history cap"
            )
    return required


def require_live_balance_support(config, override=None):
    """Reject unsupported live account-level replay before account acquisition."""
    require_live_runtime_support(config)
    mode = _mode(config)
    if mode == "coin":
        return
    value = (
        override
        if override is not None
        else config.get("live", {}).get("account", {}).get("balance_override")
    )
    bot = config.get("bot", {})
    enabled = (
        bot["portfolio"]["hsl"]["enabled"]
        if mode == "portfolio"
        else any(
            bot.get(side, {}).get("hsl", {}).get("enabled", False)
            for side in ("long", "short")
        )
    )
    if enabled and value not in (None, ""):
        raise ValueError(
            "enabled portfolio/position_side HSL does not support live.account.balance_override; remove the override, select coin mode, or disable HSL"
        )


def require_runtime_support(config, supported_modes=()):
    engine(config)
    if _mode(config) not in supported_modes:
        raise ValueError(
            f"HSL {_mode(config)} runtime integration is not available in this build"
        )


def _side_has_enabled_policy(config, side, markets_by_exchange):
    base_enabled = config["bot"][side]["hsl"]["enabled"]
    if _mode(config) != "coin":
        return base_enabled
    datasets = config.get("backtest", {}).get("coins")
    if datasets is None:
        # Config-only callers can check the declared policies before a dataset
        # exists. Runtime callers below use only their actual dataset members.
        return base_enabled or any(
            patch.get("bot", {})
            .get(side, {})
            .get("hsl", {})
            .get("enabled", base_enabled)
            for patch in config.get("coin_overrides", {}).values()
        )
    from backtest import _get_backtest_coin_override

    for exchange, coins in datasets.items():
        markets = (markets_by_exchange or {}).get(exchange, {})
        for coin in coins:
            patch = _get_backtest_coin_override(config, markets, exchange, coin)
            if (
                patch.get("bot", {})
                .get(side, {})
                .get("hsl", {})
                .get("enabled", base_enabled)
            ):
                return True
    return False


def validate_optimizer_metrics(config, metrics, *, markets_by_exchange=None):
    """Validate only objectives/limits consuming this effective scenario's results."""
    engine(config)
    from .metrics import (
        canonical_metric_name,
        canonicalize_metric_name,
        split_metric_stat_suffix,
    )

    for metric in metrics:
        name = canonicalize_metric_name(metric)
        name = name.removesuffix("_usd").removesuffix("_btc")
        name, _ = split_metric_stat_suffix(canonical_metric_name(name))
        if name.startswith(("hard_stop_time_in_yellow", "hard_stop_time_in_orange")):
            raise ValueError(
                f"{metric} is removed for HSL; choose a supported objective/limit"
            )
        signal = (
            name.startswith("hard_stop_")
            or name.startswith("drawdown_worst_ema_strategy_eq")
            or name.startswith("drawdown_worst_mean_1pct_ema_strategy_eq")
        )
        side_signal = signal and name.endswith(("_long", "_short"))
        if _mode(config) == "portfolio" and side_signal:
            raise ValueError(
                f"{metric} has no side controller in portfolio HSL; use the portfolio metric"
            )
        if side_signal and not _side_has_enabled_policy(
            config, name.rsplit("_", 1)[1], markets_by_exchange
        ):
            raise ValueError(
                f"{metric} has no enabled side controller in this HSL scenario"
            )
        if signal and not side_signal:
            enabled = (
                config["bot"]["portfolio"]["hsl"]["enabled"]
                if _mode(config) == "portfolio"
                else any(
                    _side_has_enabled_policy(config, side, markets_by_exchange)
                    for side in ("long", "short")
                )
            )
            if not enabled:
                raise ValueError(
                    f"{metric} has no enabled controller in this HSL scenario"
                )


def native_hsl_policy(policy):
    """Transport public HSL field names to the stable native HSL ABI."""
    result = deepcopy(policy)
    result["red_threshold"] = result.pop("drawdown_ratio_threshold")
    result["cooldown_minutes_after_red"] = result.pop("cooldown_after_red_minutes")
    return result


def native_hsl_mode(mode):
    """Translate a validated public scope to the native engine's mode token."""
    from .coerce import normalize_hsl_signal_mode

    mode = normalize_hsl_signal_mode(mode)
    return "pside" if mode == "position_side" else mode
