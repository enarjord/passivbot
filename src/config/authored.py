"""Validate authored configuration structure before normalization can erase input.

Defaults describe closed objects. The few dynamic mappings and record lists are
declared here; their owning validators still enforce values and semantic rules.
This boundary performs no migration and does not import replay/exchange runtimes.
"""

from copy import deepcopy
from functools import lru_cache

from .access import get_optional_config_value
from .field_registry import get_field_registry
from .field_spec import FieldRegistry, FieldType
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
_BOOLEAN_RECORD_COLLECTIONS = frozenset({
    ("optimize", "limits"), ("optimize", "prefilter", "checks"),
})


@lru_cache(maxsize=1)
def _authored_boolean_fields():
    # Boolean declarations do not depend on numeric defaults or strategy parameters.
    return FieldRegistry(
        field for field in get_field_registry() if field.types == (FieldType.BOOLEAN,)
    )


def validate_authored_boolean(value, parts, *, path):
    if type(value) is bool:
        return
    field = _authored_boolean_fields().find(parts)
    if field is None or (field.nullable and value is None):
        return
    expected = "a boolean or null" if field.nullable else "a boolean"
    raise TypeError(f"{path} must be {expected}")


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
    from .schema import get_expanded_config_prototype

    schema = get_expanded_config_prototype()
    # A referenced coin document has one existing standalone allocation leaf.
    # Preserve that explicit code-owned validation extension without accepting
    # arbitrary caller-template fields as canonical bot policy.
    from .field_policies import OVERRIDABLE_STANDALONE_BOT_KEYS
    for side in ("long", "short"):
        declared = template.get("bot", {}).get(side, {})
        for key in OVERRIDABLE_STANDALONE_BOT_KEYS:
            if key in declared:
                schema["bot"][side][key] = deepcopy(declared[key])
    policy_context = config if policy_context is None else policy_context
    # Added by the loader for relative file resolution, rather than user policy.
    schema["live"]["base_config_path"] = ""
    from .hsl import FIELDS

    portfolio_risk = schema["bot"]["portfolio"]["risk"]
    portfolio_risk["hsl"].update({key: None for key in FIELDS})
    portfolio_risk["loss_realization"]["loss_gate"]["max_realized_drawdown_fraction"] = 0.0
    bounds = schema["optimize"]["bounds"]
    for side in ("long", "short"):
        bounds[side]["risk"]["hsl"]["memory_ema_span_minutes"] = []
        cooldown = bounds[side]["entry_cooldown"]
        for key in ("min_duration_minutes", "max_duration_minutes"):
            cooldown[key] = []
        cooldown["weights_minutes"] = {"exposure_ratio": [], "adverse_directionality": []}
        bounds[side]["forager"].setdefault("score_weights", {})["rms_directionality"] = []
        bounds[side]["forager"]["rms_directionality_ema_span_minutes"] = []
    bounds["global"] = deepcopy(bounds["long"])
    # Optional dimensions must remain absent from default searches.
    risk_bounds = bounds.setdefault("portfolio", {}).setdefault("risk", {})
    risk_bounds["loss_realization"] = {"memory_ema_span_minutes": [],
                                      "loss_gate": {"max_realized_drawdown_fraction": []}}
    risk_bounds.setdefault("hsl", {}).update({
        "drawdown_threshold_fraction": [], "drawdown_fraction_ema_span_minutes": [],
        "memory_ema_span_minutes": [], "cooldown_after_red_minutes": [],
    })

    def visit(value, expected, parts, path="config"):
        from .control_sentinels import validate_control_value

        validate_control_value(parts, value)
        if parts == ("bot", "global", "enabled"):
            from .activation import GLOBAL_ACTIVATION_ERROR
            raise ValueError(GLOBAL_ACTIVATION_ERROR)
        validate_authored_boolean(value, parts, path=path)
        from .age_policy import AGE_PATHS, validate_age_value
        if parts in AGE_PATHS:
            validate_age_value(value, path=".".join(parts))
        if (len(parts) == 4 and parts[0] == "bot" and parts[1] in {"global", "long", "short"}
                and parts[2] == "universe" and parts[3] in {"approved_coins", "ignored_coins"}):
            if parts[1] == "global" and parts[3] == "approved_coins" and value is None:
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

            mode = policy_context.get("bot", {}).get("portfolio", {}).get("risk", {}).get("hsl", {}).get("scope", "coin")
            strategy = get_optional_config_value(policy_context, 'bot.portfolio.strategy_kind', "trailing_martingale")
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
        if parts in _BOOLEAN_RECORD_COLLECTIONS and isinstance(value, list):
            for index, record in enumerate(value):
                if isinstance(record, dict) and "enabled" in record:
                    validate_authored_boolean(
                        record["enabled"], (*parts, index, "enabled"), path=f"{path}[{index}].enabled"
                    )
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
                if (*parts, key) == ("bot", "global", "enabled"):
                    from .activation import GLOBAL_ACTIVATION_ERROR
                    raise ValueError(GLOBAL_ACTIVATION_ERROR)
                if key not in expected:
                    raise ValueError(f"{path}.{key} is an unknown configuration field")
                visit(child, expected[key], (*parts, key), f"{path}.{key}")

    def validate_patch(patch, path):
        _mapping(patch, path)

        def validate_group(parts, value):
            # Empty nested objects can disappear during flattening.
            validate_authored_boolean(value, parts, path=f"{path}.{'.'.join(parts)}")

        normalized = normalize_scenario_overrides(patch, validate_group=validate_group)
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
                mode = policy_context.get("bot", {}).get("portfolio", {}).get("risk", {}).get("hsl", {}).get("scope", "coin")
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
                    strategy_kind=get_optional_config_value(policy_context, 'bot.portfolio.strategy_kind', "trailing_martingale"),
                    hsl_signal_mode=mode,
                )
                continue
            if parts == ("bot", "global", "enabled"):
                from .activation import GLOBAL_ACTIVATION_ERROR
                raise ValueError(GLOBAL_ACTIVATION_ERROR)
            expected = schema
            for part in parts:
                if not isinstance(expected, dict) or part not in expected:
                    raise ValueError(f"{path}.{selector} targets an unknown configuration field")
                expected = expected[part]
            visit(value, expected, parts, f"{path}.{selector}")

    visit(config, schema, ())
