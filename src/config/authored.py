"""Validate authored configuration structure before normalization can erase input.

Defaults describe closed objects. The few dynamic mappings and record lists are
declared here; their owning validators still enforce values and semantic rules.
This boundary performs no migration and does not import replay/exchange runtimes.
"""

from copy import deepcopy

from .scenario_overrides import normalize_scenario_overrides


_METADATA_KEYS = frozenset({"_raw", "_raw_effective", "_transform_log", "_coins_sources"})
_DYNAMIC_MAPS = frozenset(
    {
        ("backtest", "coin_sources"),
        ("backtest", "market_settings_sources"),
        ("backtest", "reducer"),
        ("live", "startup", "startup_phase_budgets"),
    }
)
_SCENARIO_FIELDS = frozenset({
    "label", "start_date", "end_date", "coins", "ignored_coins", "exchanges",
    "coin_sources", "overrides",
})
_LIMIT_FIELDS = frozenset({
    "enabled", "metric", "penalize_if", "reducer", "scenario", "value", "range",
})
_SCORING_FIELDS = frozenset({"metric", "goal", "reducer", "scenario"})
_MARKET_FIELDS = frozenset({"qty_step", "price_step", "min_qty", "min_cost", "c_mult"})


from config.access import get_optional_config_value

def _mapping(value, path):
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a dict")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings")


def _closed_fields(value, allowed, path):
    _mapping(value, path)
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is an unknown configuration field")


def _coin_source(value, path, *, allow_sides=True):
    # Source strings include curated lists, the all sentinel, and file refs.
    # Resolving files belongs to the coin-source owner after structural checks.
    if value is None or isinstance(value, str):
        return
    if isinstance(value, list) and all(isinstance(coin, str) for coin in value):
        return
    if allow_sides and isinstance(value, dict):
        _closed_fields(value, {"long", "short"}, path)
        for side, source in value.items():
            _coin_source(source, f"{path}.{side}", allow_sides=False)
        return
    raise TypeError(f"{path} must be a coin list, source string, null, or long/short mapping")


def validate_authored_config(config, template, *, policy_context=None):
    """Reject unsupported authored fields, including sparse nested patches.

    Re-normalizing canonical configs is supported through the exact internal
    metadata keys. Metadata is never interpreted as authored trading authority.
    Sparse external documents may use their execution base as policy_context;
    it affects coin-patch policy, never the authored structural field checks.
    """
    _mapping(config, "config")
    schema = deepcopy(template)
    policy_context = config if policy_context is None else policy_context
    # Added by the loader for relative file resolution, rather than user policy.
    schema["live"]["base_config_path"] = ""
    from .hsl import FIELDS

    schema["bot"]["global"]["hsl"].update({key: None for key in FIELDS})
    bounds = schema["optimize"]["bounds"]
    for side in ("long", "short"):
        bounds[side]["hsl"]["memory_ema_span_minutes"] = []
        cooldown = bounds[side]["entry_cooldown"]
        for key in ("min_duration_minutes", "max_duration_minutes"):
            cooldown[key] = []
        cooldown["weights_minutes"] = {"exposure_ratio": [], "adverse_directionality": []}
        bounds[side]["forager"].setdefault("score_weights", {})["unilateralness"] = []
        bounds[side]["forager"]["unilateralness_ema_span_minutes"] = []
    # Optional dimensions must remain absent from default searches.
    bounds.setdefault("global", {}).setdefault("unstuck", {})["memory_ema_span_minutes"] = []
    bounds.setdefault("global", {}).setdefault("loss_gate", {})["memory_ema_span_minutes"] = []
    bounds["global"].setdefault("hsl", {}).update({
        "drawdown_ratio_threshold": [], "drawdown_ratio_ema_span_minutes": [],
        "memory_ema_span_minutes": [], "cooldown_after_red_minutes": [],
    })

    def visit(value, expected, parts):
        from .control_sentinels import validate_control_value

        validate_control_value(parts, value)
        path = "config" + ("." + ".".join(parts) if parts else "")
        if (len(parts) == 4 and parts[0] == "bot" and parts[1] in {"global", "long", "short"}
                and parts[2] == "universe" and parts[3] in {"approved_coins", "ignored_coins"}):
            if parts[1] == "global" and value is None:
                raise TypeError(f"{path} must be a coin list or source string; null is only side inheritance")
            if parts[1] == "global":
                text = value if isinstance(value, str) else ",".join(value) if isinstance(value, list) and all(isinstance(x, str) for x in value) else ""
                if text.strip().startswith("{") and text.strip().endswith("}"):
                    raise TypeError(f"{path} must be one common coin list, all, or file; use explicit side source paths for directional selections")
            _coin_source(value, path, allow_sides=False)
            return
        if parts == ("coin_overrides",):
            _mapping(value, path)
            from .overrides import _extract_allowed_patch

            mode = policy_context.get("bot", {}).get("global", {}).get("hsl", {}).get("scope", "coin")
            strategy = get_optional_config_value(policy_context, 'bot.global.strategy_kind', "trailing_martingale")
            for coin, patch in value.items():
                _extract_allowed_patch(
                    patch, coin=coin, source=f"coin_overrides.{coin}", strict=True,
                    strategy_kind=strategy, hsl_signal_mode=mode,
                )
            return
        if parts in _DYNAMIC_MAPS:
            _mapping(value, path)
            return
        if parts == ("backtest", "market_settings"):
            _closed_fields(value, {"overrides", "overrides_by_exchange"}, path)
            for coin, settings in value.get("overrides", {}).items():
                _closed_fields(settings, _MARKET_FIELDS, f"{path}.overrides.{coin}")
            for exchange, coins in value.get("overrides_by_exchange", {}).items():
                _mapping(coins, f"{path}.overrides_by_exchange.{exchange}")
                for coin, settings in coins.items():
                    _closed_fields(settings, _MARKET_FIELDS,
                                   f"{path}.overrides_by_exchange.{exchange}.{coin}")
            return
        if parts == ("optimize", "fixed_runtime_overrides"):
            validate_patch(value, path)
            return
        records = {
            ("backtest", "scenarios"): _SCENARIO_FIELDS,
            ("optimize", "limits"): _LIMIT_FIELDS,
            ("optimize", "scoring"): _SCORING_FIELDS,
        }
        if parts in records:
            if not isinstance(value, list):
                raise TypeError(f"{path} must be a list")
            for index, record in enumerate(value):
                record_path = f"{path}[{index}]"
                if parts == ("optimize", "scoring") and isinstance(record, str):
                    # The scoring owner validates built-in shorthand and its
                    # default goal before producing canonical objective records.
                    continue
                _closed_fields(record, records[parts], record_path)
                if parts == ("backtest", "scenarios") and "overrides" in record:
                    validate_patch(record["overrides"], f"{record_path}.overrides")
            return
        if isinstance(expected, dict):
            _mapping(value, path)
            for key, child in value.items():
                if not parts and key in _METADATA_KEYS:
                    continue
                if key not in expected:
                    raise ValueError(f"{path}.{key} is an unknown configuration field")
                visit(child, expected[key], (*parts, key))

    def validate_patch(patch, path):
        _mapping(patch, path)
        normalized = normalize_scenario_overrides(patch)
        paths = {tuple(selector.split(".")): selector for selector in normalized}
        # Assigning a complete dotted object replaces its descendants. Reject
        # multiple authorities before insertion order can choose a winner.
        for parts in sorted(paths):
            for length in range(1, len(parts)):
                ancestor = parts[:length]
                if ancestor in paths:
                    raise ValueError(
                        f"{path} has overlapping configuration paths "
                        f"{paths[ancestor]!r} and {paths[parts]!r}"
                    )
        for selector, value in normalized.items():
            # Authored patches carry configuration authority. Selector shorthand
            # belongs only to fixed_params/fine_tune bound matching.
            parts = tuple(selector.split("."))
            if parts[:1] == ("coin_overrides",) and len(parts) > 1:
                from .overrides import _extract_allowed_patch, get_allowed_modifications

                coin, *local_parts = parts[1:]
                mode = policy_context.get("bot", {}).get("global", {}).get("hsl", {}).get("scope", "coin")
                policy = get_allowed_modifications(hsl_signal_mode=mode)
                for part in local_parts:
                    if not isinstance(policy, dict) or part not in policy:
                        raise ValueError(f"{path}.{selector} is not overridable")
                    policy = policy[part]
                local_patch = value if not local_parts else {}
                if local_parts:
                    target = local_patch
                    for part in local_parts[:-1]:
                        target = target.setdefault(part, {})
                    target[local_parts[-1]] = value
                _extract_allowed_patch(
                    local_patch, coin=coin, source=f"{path}.{selector}", strict=True,
                    strategy_kind=get_optional_config_value(policy_context, 'bot.global.strategy_kind', "trailing_martingale"),
                    hsl_signal_mode=mode,
                )
                continue
            expected = schema
            for part in parts:
                if not isinstance(expected, dict) or part not in expected:
                    raise ValueError(f"{path}.{selector} targets an unknown configuration field")
                expected = expected[part]
            visit(value, expected, parts)

    visit(config, schema, ())
