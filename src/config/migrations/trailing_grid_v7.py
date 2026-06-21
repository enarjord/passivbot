from __future__ import annotations

import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.optimize_bounds import (
    BOT_BOUND_GROUP_BY_KEY,
    SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY,
    set_flat_optimize_bound,
    sort_optimize_bounds_in_place,
)
from config.overrides import allowed_flat_bot_side_modification_keys
from config.schema import CONFIG_SCHEMA_VERSION, get_template_config
from config.shared_bot import (
    BOT_GROUP_FIELD_MAP,
    FLAT_BOT_KEY_TO_GROUP_PATH,
    get_grouped_bot_value,
)
from config.strategy_spec import get_strategy_param_keys
from risk_limits import WE_EXCESS_ALLOWANCE_MODE_BOUNDED


TRAILING_GRID_V7_KIND = "trailing_grid_v7"

ENTRY_FIELD_MAP = {
    "entry_grid_double_down_factor": ("entry", "grid_double_down_factor"),
    "entry_grid_spacing_pct": ("entry", "grid_spacing_pct"),
    "entry_grid_spacing_we_weight": ("entry", "grid_spacing_we_weight"),
    "entry_grid_spacing_volatility_weight": ("entry", "grid_spacing_volatility_weight"),
    "entry_initial_ema_dist": ("entry", "initial_ema_dist"),
    "entry_initial_qty_pct": ("entry", "initial_qty_pct"),
    "entry_trailing_double_down_factor": ("entry", "trailing_double_down_factor"),
    "entry_trailing_grid_ratio": ("entry", "trailing_grid_ratio"),
    "entry_trailing_retracement_pct": ("entry", "trailing_retracement_pct"),
    "entry_trailing_retracement_we_weight": ("entry", "trailing_retracement_we_weight"),
    "entry_trailing_retracement_volatility_weight": (
        "entry",
        "trailing_retracement_volatility_weight",
    ),
    "entry_trailing_threshold_pct": ("entry", "trailing_threshold_pct"),
    "entry_trailing_threshold_we_weight": ("entry", "trailing_threshold_we_weight"),
    "entry_trailing_threshold_volatility_weight": (
        "entry",
        "trailing_threshold_volatility_weight",
    ),
    "entry_volatility_ema_span_hours": ("entry", "volatility_ema_span_hours"),
    "entry_volatility_ema_span_1h": ("entry", "volatility_ema_span_hours"),
}

CLOSE_FIELD_MAP = {
    "close_grid_markup_start": ("close", "grid_markup_start"),
    "close_grid_markup_end": ("close", "grid_markup_end"),
    "close_grid_qty_pct": ("close", "grid_qty_pct"),
    "close_trailing_grid_ratio": ("close", "trailing_grid_ratio"),
    "close_trailing_qty_pct": ("close", "trailing_qty_pct"),
    "close_trailing_retracement_pct": ("close", "trailing_retracement_pct"),
    "close_trailing_threshold_pct": ("close", "trailing_threshold_pct"),
}

ROOT_STRATEGY_FIELD_MAP = {
    "ema_span_0": ("ema_span_0",),
    "ema_span_1": ("ema_span_1",),
}

STRATEGY_FIELD_MAP = {
    **ENTRY_FIELD_MAP,
    **CLOSE_FIELD_MAP,
    **ROOT_STRATEGY_FIELD_MAP,
}

V7_DISTINCTIVE_STRATEGY_KEYS = set(ENTRY_FIELD_MAP) | set(CLOSE_FIELD_MAP)

LEGACY_BOUND_ALIASES = {
    "entry_grid_spacing_weight": "entry_grid_spacing_we_weight",
    "entry_grid_spacing_log_span_hours": "entry_volatility_ema_span_hours",
    "entry_log_range_ema_span_hours": "entry_volatility_ema_span_hours",
    "entry_grid_spacing_log_weight": "entry_grid_spacing_volatility_weight",
    "entry_trailing_retracement_log_weight": "entry_trailing_retracement_volatility_weight",
    "entry_trailing_threshold_log_weight": "entry_trailing_threshold_volatility_weight",
    "entry_volatility_ema_span_1h": "entry_volatility_ema_span_hours",
    "filter_volatility_ema_span_1m": "forager_volatility_ema_span_1m",
    "filter_volatility_ema_span": "forager_volatility_ema_span_1m",
    "filter_noisiness_rolling_window": "forager_volatility_ema_span_1m",
    "filter_noisiness_ema_span": "forager_volatility_ema_span_1m",
    "filter_log_range_ema_span": "forager_volatility_ema_span_1m",
    "filter_volume_ema_span_1m": "forager_volume_ema_span_1m",
    "filter_volume_ema_span": "forager_volume_ema_span_1m",
    "filter_volume_rolling_window": "forager_volume_ema_span_1m",
    "filter_volume_drop_pct": "forager_volume_drop_pct",
    "forager_volatility_ema_span": "forager_volatility_ema_span_1m",
    "forager_volume_ema_span": "forager_volume_ema_span_1m",
}

OBSOLETE_BOUND_KEYS = {
    "filter_volatility_drop_pct",
}

MANUAL_REVIEW_BOUND_KEYS = {
    "min_markup",
    "close_grid_min_markup",
    "markup_range",
    "close_grid_markup_range",
}

COIN_OVERRIDE_SIDE_PASSTHROUGH_KEYS = {
    "wallet_exposure_limit",
}
COIN_OVERRIDE_SUPPORTED_SHARED_FLAT_KEYS = allowed_flat_bot_side_modification_keys()

V7_ABSENT_RISK_DEFAULTS = {
    "risk_entry_cooldown_minutes": 0.0,
    "risk_we_excess_allowance_mode": WE_EXCESS_ALLOWANCE_MODE_BOUNDED,
}
V7_INSERTED_DEFAULT_TOP_LEVEL_PATHS = (
    ("backtest", "candle_interval_minutes"),
    ("backtest", "dynamic_wel_by_tradability"),
    ("backtest", "exchanges"),
    ("backtest", "liquidation_threshold"),
    ("backtest", "maker_fee_override"),
    ("backtest", "market_order_slippage_pct"),
    ("backtest", "scenarios"),
    ("backtest", "starting_balance"),
    ("backtest", "taker_fee_override"),
    ("live", "approved_coins"),
    ("live", "forager_score_hysteresis_pct"),
    ("live", "hsl_position_during_cooldown_policy"),
    ("live", "hsl_signal_mode"),
)
V7_INSERTED_DEFAULT_SHARED_FLAT_KEYS = (
    "risk_entry_cooldown_minutes",
    "risk_twel_enforcer_enabled",
    "risk_twel_enforcer_threshold",
    "risk_wel_enforcer_enabled",
    "risk_wel_enforcer_threshold",
    "risk_we_excess_allowance_mode",
    "forager_score_weights",
    "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct",
    "forager_volume_ema_span_1m",
    "hsl_cooldown_minutes_after_red",
    "hsl_ema_span_minutes",
    "hsl_enabled",
    "hsl_no_restart_drawdown_threshold",
    "hsl_orange_tier_mode",
    "hsl_panic_close_order_type",
    "hsl_red_threshold",
    "hsl_tier_ratios",
    "unstuck_close_pct",
    "unstuck_ema_dist",
    "unstuck_enabled",
    "unstuck_loss_allowance_pct",
    "unstuck_threshold",
)
V7_THRESHOLD_DERIVED_ENFORCER_ENABLED_KEYS = {
    "risk_wel_enforcer_enabled",
    "risk_twel_enforcer_enabled",
}
V7_ENFORCER_THRESHOLD_BY_ENABLED_KEY = {
    "risk_wel_enforcer_enabled": "risk_wel_enforcer_threshold",
    "risk_twel_enforcer_enabled": "risk_twel_enforcer_threshold",
}

LEGACY_SHARED_SIDE_ALIASES = {
    "filter_volatility_ema_span_1m": "forager_volatility_ema_span_1m",
    "filter_volatility_ema_span": "forager_volatility_ema_span_1m",
    "filter_noisiness_rolling_window": "forager_volatility_ema_span_1m",
    "filter_noisiness_ema_span": "forager_volatility_ema_span_1m",
    "filter_log_range_ema_span": "forager_volatility_ema_span_1m",
    "filter_volume_ema_span_1m": "forager_volume_ema_span_1m",
    "filter_volume_ema_span": "forager_volume_ema_span_1m",
    "filter_volume_rolling_window": "forager_volume_ema_span_1m",
    "filter_volume_drop_pct": "forager_volume_drop_pct",
}


AUTHORITATIVE_FORAGER_ALIAS_KEYS = {
    "forager_volatility_ema_span",
    "forager_volatility_ema_span_1m",
    "forager_volume_drop_pct",
    "forager_volume_ema_span",
    "forager_volume_ema_span_1m",
}


def _set_path(mapping: dict, path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = deepcopy(value)


def _iter_leaf_items(mapping: dict, prefix: tuple[str, ...] = ()):
    for key, value in mapping.items():
        path = (*prefix, str(key))
        if isinstance(value, dict):
            yield from _iter_leaf_items(value, path)
        else:
            yield path, value


@lru_cache(maxsize=1)
def _supported_strategy_leaf_paths() -> frozenset[tuple[str, ...]]:
    return frozenset(
        tuple(key.split("."))
        for key in get_strategy_param_keys(TRAILING_GRID_V7_KIND)
    )


def _is_supported_strategy_leaf_path(path: tuple[str, ...]) -> bool:
    return path in _supported_strategy_leaf_paths()


def _canonical_strategy_leaf_path(path: tuple[str, ...]) -> tuple[str, ...] | None:
    if _is_supported_strategy_leaf_path(path):
        return path
    flat_key = "_".join(path)
    if flat_key in OBSOLETE_BOUND_KEYS or flat_key in MANUAL_REVIEW_BOUND_KEYS:
        return None
    canonical_flat_key = LEGACY_BOUND_ALIASES.get(flat_key, flat_key)
    if canonical_flat_key in OBSOLETE_BOUND_KEYS or canonical_flat_key in MANUAL_REVIEW_BOUND_KEYS:
        return None
    mapped_path = STRATEGY_FIELD_MAP.get(canonical_flat_key)
    if mapped_path is not None and _is_supported_strategy_leaf_path(mapped_path):
        return mapped_path
    return None


def _path_leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _shared_alias_source_is_authoritative(source_path: str, canonical_flat_key: str) -> bool:
    source_key = _path_leaf(source_path)
    return source_key == canonical_flat_key or source_key in AUTHORITATIVE_FORAGER_ALIAS_KEYS


def _bound_source_key(source_path: str) -> str:
    return source_path.removeprefix("optimize.bounds.")


def _bound_local_key(flat_key: str) -> str:
    if flat_key.startswith(("long_", "short_")):
        return flat_key.split("_", 1)[1]
    return flat_key


def _bound_alias_source_is_authoritative(source_path: str, canonical_key: str) -> bool:
    source_key = _bound_source_key(source_path)
    return source_key == canonical_key or _bound_local_key(source_key) in AUTHORITATIVE_FORAGER_ALIAS_KEYS


def _is_old_filter_bound_alias(flat_key: str) -> bool:
    return _bound_local_key(flat_key).startswith("filter_")


def _bound_alias_priority(source_key: str, canonical_key: str) -> int:
    if source_key == canonical_key:
        return 0
    if _bound_local_key(source_key) in AUTHORITATIVE_FORAGER_ALIAS_KEYS:
        return 1
    return 2


def _record_target_value(
    records: dict[tuple[str, ...], tuple[str, str, Any]],
    target_key: tuple[str, ...],
    *,
    source_path: str,
    target_path: str,
    value: Any,
    report: dict,
) -> bool:
    existing = records.get(target_key)
    if existing is None:
        records[target_key] = (source_path, target_path, deepcopy(value))
        return True
    existing_source, existing_target, existing_value = existing
    if existing_value != value:
        report["manual_review_fields"].append(
            f"{source_path} conflicts with {existing_source} for {existing_target}; "
            f"kept {existing_source}"
        )
    return False


def _copy_supported_top_level_sections(source: dict, target: dict, report: dict) -> None:
    for section in ("live", "backtest", "optimize", "logging", "monitor"):
        source_section = source.get(section)
        target_section = target.get(section)
        if not isinstance(source_section, dict) or not isinstance(target_section, dict):
            continue
        for key, value in source_section.items():
            if section == "optimize" and key == "bounds":
                continue
            if key in target_section:
                target_section[key] = deepcopy(value)
                report["moved_fields"].append(f"{section}.{key}")
            else:
                report["manual_review_fields"].append(f"{section}.{key}")


def _report_unknown_top_level_sections(source: dict, report: dict) -> None:
    handled = {
        "backtest",
        "bot",
        "coin_overrides",
        "config_version",
        "live",
        "logging",
        "monitor",
        "optimize",
    }
    for key in sorted(source):
        if isinstance(key, str) and key.startswith("_"):
            continue
        if key not in handled:
            report["manual_review_fields"].append(str(key))


def migration_report_has_unresolved(report: dict) -> bool:
    return bool(
        report.get("manual_review_fields")
        or report.get("dropped_unsupported_fields")
    )


def _source_side_has_shared_value(source_side: dict, flat_key: str) -> bool:
    if flat_key in source_side:
        return True
    group_path = FLAT_BOT_KEY_TO_GROUP_PATH.get(flat_key)
    if group_path is None:
        return False
    group_name, local_key = group_path
    group = source_side.get(group_name)
    return isinstance(group, dict) and local_key in group


def _source_side_has_nonpositive_enforcer_threshold(
    source_side: dict, enabled_flat_key: str
) -> bool:
    threshold_flat_key = V7_ENFORCER_THRESHOLD_BY_ENABLED_KEY.get(enabled_flat_key)
    if threshold_flat_key is None or not _source_side_has_shared_value(
        source_side,
        threshold_flat_key,
    ):
        return False
    raw_threshold = get_grouped_bot_value(source_side, threshold_flat_key)
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return False
    return math.isfinite(threshold) and threshold <= 0.0


def _source_has_shared_bound(source: dict, pside: str, flat_key: str) -> bool:
    bounds = source.get("optimize", {}).get("bounds")
    if not isinstance(bounds, dict):
        return False
    flat_bound_key = f"{pside}_{flat_key}"
    if flat_bound_key in bounds:
        return True
    side_bounds = bounds.get(pside)
    if not isinstance(side_bounds, dict):
        return False
    group_path = FLAT_BOT_KEY_TO_GROUP_PATH.get(flat_key)
    if group_path is None:
        return False
    group_name, local_key = group_path
    group_bounds = side_bounds.get(group_name)
    return isinstance(group_bounds, dict) and local_key in group_bounds


def _path_exists(mapping: dict, path: tuple[str, ...]) -> bool:
    current = mapping
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _append_inserted_default(report: dict, path: str) -> None:
    inserted = report.setdefault("inserted_v8_defaults", [])
    if path not in inserted:
        inserted.append(path)


def _source_side_has_strategy_path(source_side: dict, strategy_path: tuple[str, ...]) -> bool:
    return any(
        legacy_key in source_side and path == strategy_path
        for legacy_key, path in STRATEGY_FIELD_MAP.items()
    )


def _flat_bound_target_path(pside: str, flat_key: str) -> tuple[str, ...] | None:
    prefix = f"{pside}_"
    if not isinstance(flat_key, str) or not flat_key.startswith(prefix):
        return None
    local_key = flat_key[len(prefix) :]
    group_name = BOT_BOUND_GROUP_BY_KEY.get(local_key)
    if group_name is not None:
        local_map = SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY[group_name]
        for target_local_key, candidate_flat_key in local_map.items():
            if candidate_flat_key == local_key:
                return ("optimize", "bounds", pside, group_name, target_local_key)
    strategy_path = STRATEGY_FIELD_MAP.get(local_key)
    if strategy_path is not None:
        return (
            "optimize",
            "bounds",
            pside,
            "strategy",
            TRAILING_GRID_V7_KIND,
            *strategy_path,
        )
    return None


def _nested_bound_target_path(
    pside: str,
    group_name: str,
    leaf_path: tuple[str, ...],
) -> tuple[str, ...] | None:
    if group_name == "strategy":
        canonical_path = _canonical_strategy_leaf_path(leaf_path)
        if canonical_path is None:
            return None
        return (
            "optimize",
            "bounds",
            pside,
            "strategy",
            TRAILING_GRID_V7_KIND,
            *canonical_path,
        )
    supported_local_keys = SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.get(group_name)
    if supported_local_keys is None:
        return None
    flat_local_key = "_".join(leaf_path)
    if flat_local_key not in supported_local_keys:
        return None
    return ("optimize", "bounds", pside, group_name, flat_local_key)


def _source_has_optimize_bound_path(source: dict, target_path: tuple[str, ...]) -> bool:
    if len(target_path) < 4 or target_path[:2] != ("optimize", "bounds"):
        return False
    pside = target_path[2]
    bounds = source.get("optimize", {}).get("bounds")
    if not isinstance(bounds, dict):
        return False

    for key in bounds:
        if not isinstance(key, str):
            continue
        canonical_key = _legacy_bound_key(pside, key)
        if canonical_key is None:
            continue
        if _flat_bound_target_path(pside, canonical_key) == target_path:
            return True

    side_bounds = bounds.get(pside)
    if not isinstance(side_bounds, dict):
        return False
    for group_name, group_bounds in side_bounds.items():
        if not isinstance(group_bounds, dict):
            continue
        if group_name == "strategy":
            for strategy_name, legacy_strategy in group_bounds.items():
                if strategy_name not in {"trailing_grid", TRAILING_GRID_V7_KIND}:
                    continue
                if not isinstance(legacy_strategy, dict):
                    continue
                for leaf_path, _value in _iter_leaf_items(legacy_strategy):
                    if (
                        _nested_bound_target_path(
                            pside,
                            "strategy",
                            leaf_path,
                        )
                        == target_path
                    ):
                        return True
            continue
        for leaf_path, _value in _iter_leaf_items(group_bounds):
            if _nested_bound_target_path(pside, str(group_name), leaf_path) == target_path:
                return True
    return False


def _source_side_has_default_source_value(source_side: dict, flat_key: str) -> bool:
    if _source_side_has_shared_value(source_side, flat_key):
        return True
    if flat_key == "hsl_tier_ratios":
        return (
            "hsl_tier_ratio_yellow" in source_side
            or "hsl_tier_ratio_orange" in source_side
        )
    return any(
        legacy_key in source_side
        for legacy_key, canonical_key in LEGACY_SHARED_SIDE_ALIASES.items()
        if canonical_key == flat_key
    )


def _record_inserted_v8_defaults(source: dict, target: dict, report: dict) -> None:
    for path in V7_INSERTED_DEFAULT_TOP_LEVEL_PATHS:
        if not _path_exists(source, path) and _path_exists(target, path):
            _append_inserted_default(report, ".".join(path))

    source_bot = source.get("bot", {})
    target_bot = target.get("bot", {})
    for pside in ("long", "short"):
        source_side = source_bot.get(pside) if isinstance(source_bot, dict) else None
        if not isinstance(source_side, dict):
            source_side = {}
        for flat_key in V7_INSERTED_DEFAULT_SHARED_FLAT_KEYS:
            group_path = FLAT_BOT_KEY_TO_GROUP_PATH.get(flat_key)
            if group_path is None:
                continue
            group_name, local_key = group_path
            target_path = ("bot", pside, group_name, local_key)
            if _source_side_has_default_source_value(source_side, flat_key):
                continue
            if (
                flat_key in V7_THRESHOLD_DERIVED_ENFORCER_ENABLED_KEYS
                and _source_side_has_nonpositive_enforcer_threshold(source_side, flat_key)
            ):
                continue
            if _path_exists({"bot": target_bot}, target_path):
                _append_inserted_default(report, ".".join(target_path))

        target_strategy = (
            target_bot
            .get(pside, {})
            .get("strategy", {})
            .get(TRAILING_GRID_V7_KIND, {})
        )
        if isinstance(target_strategy, dict):
            for strategy_path, _value in _iter_leaf_items(target_strategy):
                if _source_side_has_strategy_path(source_side, strategy_path):
                    continue
                _append_inserted_default(
                    report,
                    ".".join(("bot", pside, "strategy", TRAILING_GRID_V7_KIND, *strategy_path)),
                )

    target_bounds = target.get("optimize", {}).get("bounds", {})
    if isinstance(target_bounds, dict):
        for pside in ("long", "short"):
            side_bounds = target_bounds.get(pside, {})
            if not isinstance(side_bounds, dict):
                continue
            for bound_path, _value in _iter_leaf_items(side_bounds):
                target_path = ("optimize", "bounds", pside, *bound_path)
                if _source_has_optimize_bound_path(source, target_path):
                    continue
                _append_inserted_default(report, ".".join(target_path))

    inserted = report.setdefault("inserted_v8_defaults", [])
    if inserted:
        examples = ", ".join(inserted[:8])
        suffix = "" if len(inserted) <= 8 else f", ... ({len(inserted)} total)"
        report["warnings"].append(
            "v7 migration inserted v8 default values for fields absent from the source config; "
            f"review report.inserted_v8_defaults before running live/large optimizations: "
            f"{examples}{suffix}."
        )


def _force_v7_absent_risk_defaults(source: dict, target: dict, report: dict) -> None:
    bot = source.get("bot", {})
    for pside in ("long", "short"):
        source_side = bot.get(pside, {}) if isinstance(bot, dict) else {}
        if not isinstance(source_side, dict):
            source_side = {}
        target_risk = target["bot"][pside].setdefault("risk", {})
        target_bounds_risk = target["optimize"]["bounds"][pside].setdefault("risk", {})
        for flat_key, default_value in V7_ABSENT_RISK_DEFAULTS.items():
            group_name, local_key = FLAT_BOT_KEY_TO_GROUP_PATH[flat_key]
            if group_name != "risk":
                continue
            if _source_side_has_shared_value(source_side, flat_key):
                value = target_risk.get(local_key)
            else:
                value = deepcopy(default_value)
                old_value = target_risk.get(local_key)
                target_risk[local_key] = value
                if old_value != value:
                    report["warnings"].append(
                        f"bot.{pside}.risk.{local_key} was not a v7 parameter; "
                        f"using {value!r} for v7 behavior instead of the v8 template value "
                        f"{old_value!r}."
                    )
            if flat_key == "risk_entry_cooldown_minutes" and not _source_has_shared_bound(
                source, pside, flat_key
            ):
                target_bounds_risk[local_key] = [float(value), float(value), 0.1]


def _disable_enforcers_for_zero_v7_thresholds(source: dict, target: dict, report: dict) -> None:
    bot = source.get("bot", {})
    if not isinstance(bot, dict):
        return
    enforcer_pairs = (
        (
            "risk_wel_enforcer_threshold",
            "position_exposure_enforcer_threshold",
            "position_exposure_enforcer_enabled",
        ),
        (
            "risk_twel_enforcer_threshold",
            "total_exposure_enforcer_threshold",
            "total_exposure_enforcer_enabled",
        ),
    )
    for pside in ("long", "short"):
        source_side = bot.get(pside, {})
        if not isinstance(source_side, dict):
            continue
        target_risk = target["bot"][pside].setdefault("risk", {})
        for flat_threshold_key, threshold_key, enabled_key in enforcer_pairs:
            if not _source_side_has_shared_value(source_side, flat_threshold_key):
                continue
            raw_threshold = get_grouped_bot_value(source_side, flat_threshold_key)
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(threshold) or threshold > 0.0:
                continue
            if target_risk.get(enabled_key) is False:
                continue
            target_risk[enabled_key] = False
            report["warnings"].append(
                f"bot.{pside}.risk.{enabled_key} set false because source "
                f"bot.{pside}.{flat_threshold_key}={raw_threshold!r} maps to "
                f"bot.{pside}.risk.{threshold_key}={raw_threshold!r}; v8 requires disabled "
                "enforcers when thresholds are zero or negative."
            )


def _warn_if_risk_excess_would_be_clamped(risk: dict, *, path: str, report: dict) -> None:
    try:
        twel = float(risk.get("total_wallet_exposure_limit", 0.0) or 0.0)
        n_positions = int(round(float(risk.get("n_positions", 0.0) or 0.0)))
        excess = float(risk.get("we_excess_allowance_pct", 0.0) or 0.0)
        explicit_wel = (
            None
            if risk.get("wallet_exposure_limit") is None
            else float(risk.get("wallet_exposure_limit") or 0.0)
        )
    except (TypeError, ValueError):
        return
    if twel <= 0.0 or n_positions <= 0 or excess <= 0.0:
        return
    base_wel = explicit_wel if explicit_wel is not None and explicit_wel > 0.0 else twel / n_positions
    raw_allowed_wel = base_wel * (1.0 + excess)
    if raw_allowed_wel <= twel:
        return
    bounded_excess = max(0.0, twel / base_wel - 1.0)
    report["warnings"].append(
        f"{path}.we_excess_allowance_pct={excess:g} would give v7 raw per-position "
        f"WEL {raw_allowed_wel:g}, above side TWEL {twel:g} "
        f"(base WEL = {base_wel:g}). The migrated v8 config keeps "
        f"{path}.we_excess_allowance_mode='bounded', so the effective excess allowance is "
        f"capped at {bounded_excess:g}. To intentionally use v7 raw/unclamped behavior, set "
        f"{path}.we_excess_allowance_mode='legacy_raw' after migration and review the added "
        f"risk explicitly."
    )


def _warn_if_v7_excess_would_be_clamped(target: dict, report: dict) -> None:
    base_risks = {
        pside: target["bot"][pside].get("risk", {})
        for pside in ("long", "short")
        if isinstance(target.get("bot", {}).get(pside, {}).get("risk"), dict)
    }
    for pside, risk in base_risks.items():
        _warn_if_risk_excess_would_be_clamped(
            risk,
            path=f"bot.{pside}.risk",
            report=report,
        )

    coin_overrides = target.get("coin_overrides")
    if not isinstance(coin_overrides, dict):
        return
    for coin, override in coin_overrides.items():
        bot = override.get("bot") if isinstance(override, dict) else None
        if not isinstance(bot, dict):
            continue
        for pside in ("long", "short"):
            override_side = bot.get(pside)
            if not isinstance(override_side, dict):
                continue
            override_risk = override_side.get("risk")
            if not isinstance(override_risk, dict):
                continue
            merged_risk = deepcopy(base_risks.get(pside, {}))
            merged_risk.update(override_risk)
            if "wallet_exposure_limit" in override_side:
                merged_risk["wallet_exposure_limit"] = override_side["wallet_exposure_limit"]
            _warn_if_risk_excess_would_be_clamped(
                merged_risk,
                path=f"coin_overrides.{coin}.bot.{pside}.risk",
                report=report,
            )


def _move_shared_side_fields(
    source_side: dict,
    target_side: dict,
    pside: str,
    report: dict,
    *,
    source_prefix: str | None = None,
    target_prefix: str | None = None,
    allowed_flat_keys: frozenset[str] | None = None,
) -> None:
    source_prefix = source_prefix or f"bot.{pside}"
    target_prefix = target_prefix or f"bot.{pside}"
    written_targets: dict[tuple[str, ...], tuple[str, str, Any]] = {}

    def _disallow_if_needed(flat_key: str, source_path: str) -> bool:
        if allowed_flat_keys is not None and flat_key not in allowed_flat_keys:
            report["manual_review_fields"].append(source_path)
            return True
        return False

    for flat_key, (group_name, local_key) in FLAT_BOT_KEY_TO_GROUP_PATH.items():
        if flat_key not in source_side:
            continue
        source_path = f"{source_prefix}.{flat_key}"
        if _disallow_if_needed(flat_key, source_path):
            continue
        value = source_side[flat_key]
        target_path = f"{target_prefix}.{group_name}.{local_key}"
        if not _record_target_value(
            written_targets,
            (group_name, local_key),
            source_path=source_path,
            target_path=target_path,
            value=value,
            report=report,
        ):
            continue
        target_side.setdefault(group_name, {})[local_key] = deepcopy(value)
        report["moved_fields"].append(
            f"{source_path} -> {target_path}"
        )

    for legacy_key, local_key in (
        ("hsl_tier_ratio_yellow", "tier_ratios.yellow"),
        ("hsl_tier_ratio_orange", "tier_ratios.orange"),
    ):
        if legacy_key not in source_side:
            continue
        value = source_side[legacy_key]
        source_path = f"{source_prefix}.{legacy_key}"
        if _disallow_if_needed("hsl_tier_ratios", source_path):
            continue
        target_path = f"{target_prefix}.hsl.{local_key}"
        path_parts = tuple(local_key.split("."))
        if not _record_target_value(
            written_targets,
            ("hsl", *path_parts),
            source_path=source_path,
            target_path=target_path,
            value=value,
            report=report,
        ):
            continue
        _set_path(target_side.setdefault("hsl", {}), path_parts, value)
        report["moved_fields"].append(
            f"{source_path} -> {target_path}"
        )

    if isinstance(source_side.get("forager_score_weights"), dict):
        value = source_side["forager_score_weights"]
        source_path = f"{source_prefix}.forager_score_weights"
        if not _disallow_if_needed("forager_score_weights", source_path):
            target_path = f"{target_prefix}.forager.score_weights"
            if _record_target_value(
                written_targets,
                ("forager", "score_weights"),
                source_path=source_path,
                target_path=target_path,
                value=value,
                report=report,
            ):
                target_side.setdefault("forager", {})["score_weights"] = deepcopy(value)
                report["moved_fields"].append(f"{source_path} -> {target_path}")

    for legacy_key, local_key in (
        ("forager_volatility_ema_span", "volatility_ema_span_1m"),
        ("forager_volume_ema_span", "volume_ema_span_1m"),
    ):
        if legacy_key not in source_side:
            continue
        value = source_side[legacy_key]
        source_path = f"{source_prefix}.{legacy_key}"
        canonical_flat_key = LEGACY_BOUND_ALIASES[legacy_key]
        if _disallow_if_needed(canonical_flat_key, source_path):
            continue
        target_path = f"{target_prefix}.forager.{local_key}"
        if not _record_target_value(
            written_targets,
            ("forager", local_key),
            source_path=source_path,
            target_path=target_path,
            value=value,
            report=report,
        ):
            continue
        target_side.setdefault("forager", {})[local_key] = deepcopy(value)
        report["moved_fields"].append(
            f"{source_path} -> {target_path}"
        )

    for legacy_key, canonical_flat_key in LEGACY_SHARED_SIDE_ALIASES.items():
        if legacy_key not in source_side:
            continue
        group_name, local_key = FLAT_BOT_KEY_TO_GROUP_PATH[canonical_flat_key]
        value = source_side[legacy_key]
        source_path = f"{source_prefix}.{legacy_key}"
        if _disallow_if_needed(canonical_flat_key, source_path):
            continue
        target_path = f"{target_prefix}.{group_name}.{local_key}"
        target_key = (group_name, local_key)
        existing = written_targets.get(target_key)
        if existing is not None and _shared_alias_source_is_authoritative(
            existing[0], canonical_flat_key
        ):
            continue
        if not _record_target_value(
            written_targets,
            target_key,
            source_path=source_path,
            target_path=target_path,
            value=value,
            report=report,
        ):
            continue
        target_side.setdefault(group_name, {})[local_key] = deepcopy(value)
        report["moved_fields"].append(
            f"{source_path} -> {target_path}"
        )


def _move_strategy_side_fields(
    source_side: dict,
    target_side: dict,
    pside: str,
    report: dict,
    *,
    source_prefix: str | None = None,
    target_prefix: str | None = None,
) -> bool:
    source_prefix = source_prefix or f"bot.{pside}"
    target_prefix = target_prefix or f"bot.{pside}"
    strategy = target_side.setdefault("strategy", {}).setdefault(TRAILING_GRID_V7_KIND, {})
    moved_any = False
    written_targets: dict[tuple[str, ...], tuple[str, str, Any]] = {}
    for legacy_key, path in STRATEGY_FIELD_MAP.items():
        if legacy_key not in source_side:
            continue
        value = source_side[legacy_key]
        source_path = f"{source_prefix}.{legacy_key}"
        target_path = (
            f"{target_prefix}.strategy.{TRAILING_GRID_V7_KIND}.{'.'.join(path)}"
        )
        if not _record_target_value(
            written_targets,
            path,
            source_path=source_path,
            target_path=target_path,
            value=value,
            report=report,
        ):
            continue
        _set_path(strategy, path, value)
        report["moved_fields"].append(
            f"{source_path} -> {target_path}"
        )
        moved_any = True
    return moved_any


def _prune_strategy_store(target: dict) -> None:
    for pside in ("long", "short"):
        side = target["bot"][pside]
        strategy = side.setdefault("strategy", {})
        active = deepcopy(strategy.get(TRAILING_GRID_V7_KIND, {}))
        side["strategy"] = {TRAILING_GRID_V7_KIND: active}
        bounds_strategy = target["optimize"]["bounds"][pside].setdefault("strategy", {})
        active_bounds = deepcopy(bounds_strategy.get(TRAILING_GRID_V7_KIND, {}))
        bounds_strategy.clear()
        bounds_strategy[TRAILING_GRID_V7_KIND] = active_bounds


def _legacy_bound_key(pside: str, key: str) -> str | None:
    prefix = f"{pside}_"
    if not key.startswith(prefix):
        return key
    local = key[len(prefix) :]
    if local in OBSOLETE_BOUND_KEYS:
        return None
    for alias, canonical in LEGACY_BOUND_ALIASES.items():
        if local == alias:
            return f"{pside}_{canonical}"
    return key


def _is_supported_flat_bound_key(key: str) -> bool:
    if not isinstance(key, str) or not key.startswith(("long_", "short_")):
        return False
    _pside, local = key.split("_", 1)
    if local in BOT_BOUND_GROUP_BY_KEY:
        return True
    path = STRATEGY_FIELD_MAP.get(local)
    return path is not None and _is_supported_strategy_leaf_path(path)


def _move_nested_strategy_bounds(
    *,
    pside: str,
    strategy_name: str,
    legacy_strategy: dict,
    target_bounds: dict,
    report: dict,
    written_targets: dict[tuple[str, ...], tuple[str, str, Any]],
) -> bool:
    moved_any = False
    target_strategy = (
        target_bounds
        .setdefault(pside, {})
        .setdefault("strategy", {})
        .setdefault(TRAILING_GRID_V7_KIND, {})
    )
    for path, value in _iter_leaf_items(legacy_strategy):
        source_path = f"optimize.bounds.{pside}.strategy.{strategy_name}.{'.'.join(path)}"
        canonical_path = _canonical_strategy_leaf_path(path)
        if canonical_path is None:
            report["manual_review_fields"].append(source_path)
            continue
        target_path = (
            f"optimize.bounds.{pside}.strategy.{TRAILING_GRID_V7_KIND}."
            f"{'.'.join(canonical_path)}"
        )
        if not _record_target_value(
            written_targets,
            canonical_path,
            source_path=source_path,
            target_path=target_path,
            value=value,
            report=report,
        ):
            continue
        _set_path(target_strategy, canonical_path, value)
        report["moved_fields"].append(
            f"{source_path} -> {target_path}"
        )
        moved_any = True
    return moved_any


def _move_nested_shared_bounds(
    *,
    pside: str,
    group_name: str,
    group_bounds: Any,
    target_bounds: dict,
    report: dict,
) -> None:
    source_prefix = f"optimize.bounds.{pside}.{group_name}"
    supported_local_keys = SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.get(group_name)
    if supported_local_keys is None or not isinstance(group_bounds, dict):
        if isinstance(group_bounds, dict):
            for path, _value in _iter_leaf_items(group_bounds):
                report["manual_review_fields"].append(f"{source_prefix}.{'.'.join(path)}")
        else:
            report["manual_review_fields"].append(source_prefix)
        return
    target_group = target_bounds.setdefault(pside, {}).setdefault(group_name, {})
    for path, value in _iter_leaf_items(group_bounds):
        source_path = f"{source_prefix}.{'.'.join(path)}"
        if len(path) != 1 or path[0] not in supported_local_keys:
            report["manual_review_fields"].append(source_path)
            continue
        local_key = path[0]
        target_group[local_key] = deepcopy(value)
        report["moved_fields"].append(f"{source_path} -> {source_path}")


def _move_bounds(source: dict, target: dict, report: dict) -> None:
    bounds = source.get("optimize", {}).get("bounds")
    if not isinstance(bounds, dict):
        return
    target_bounds = target.setdefault("optimize", {}).setdefault("bounds", {})
    if any(isinstance(key, str) and key.startswith(("long_", "short_")) for key in bounds):
        bound_items: list[tuple[int, int, str, str, Any]] = []
        for index, (key, value) in enumerate(bounds.items()):
            if not isinstance(key, str) or not key.startswith(("long_", "short_")):
                report["manual_review_fields"].append(f"optimize.bounds.{key}")
                continue
            pside = key.split("_", 1)[0]
            local_key = key[len(pside) + 1 :]
            if local_key in MANUAL_REVIEW_BOUND_KEYS:
                report["manual_review_fields"].append(f"optimize.bounds.{key}")
                continue
            canonical_key = _legacy_bound_key(pside, key)
            if canonical_key is None:
                report["dropped_unsupported_fields"].append(f"optimize.bounds.{key}")
                continue
            if not _is_supported_flat_bound_key(canonical_key):
                report["manual_review_fields"].append(f"optimize.bounds.{key}")
                continue
            priority = _bound_alias_priority(key, canonical_key)
            bound_items.append((priority, index, key, canonical_key, value))
        written_targets: dict[tuple[str, ...], tuple[str, str, Any]] = {}
        for _priority, _index, key, canonical_key, value in sorted(bound_items):
            source_path = f"optimize.bounds.{key}"
            target_path = f"optimize.bounds.{canonical_key}"
            target_key = (canonical_key,)
            existing = written_targets.get(target_key)
            if (
                existing is not None
                and _is_old_filter_bound_alias(key)
                and _bound_alias_source_is_authoritative(existing[0], canonical_key)
            ):
                continue
            if not _record_target_value(
                written_targets,
                target_key,
                source_path=source_path,
                target_path=target_path,
                value=value,
                report=report,
            ):
                continue
            set_flat_optimize_bound(target_bounds, TRAILING_GRID_V7_KIND, canonical_key, value)
            report["moved_fields"].append(
                f"optimize.bounds.{key} -> optimize.bounds.{canonical_key}"
            )
    else:
        for pside in ("long", "short"):
            side_bounds = bounds.get(pside)
            if not isinstance(side_bounds, dict):
                continue
            for group_name, group_bounds in side_bounds.items():
                if group_name != "strategy":
                    _move_nested_shared_bounds(
                        pside=pside,
                        group_name=group_name,
                        group_bounds=group_bounds,
                        target_bounds=target_bounds,
                        report=report,
                    )
                    continue
                if isinstance(group_bounds, dict):
                    handled_strategy_names = {"trailing_grid", TRAILING_GRID_V7_KIND}
                    written_strategy_targets: dict[tuple[str, ...], tuple[str, str, Any]] = {}
                    for strategy_name in ("trailing_grid", TRAILING_GRID_V7_KIND):
                        legacy_strategy = group_bounds.get(strategy_name)
                        if isinstance(legacy_strategy, dict):
                            _move_nested_strategy_bounds(
                                pside=pside,
                                strategy_name=strategy_name,
                                legacy_strategy=legacy_strategy,
                                target_bounds=target_bounds,
                                report=report,
                                written_targets=written_strategy_targets,
                            )
                        elif legacy_strategy is not None:
                            report["manual_review_fields"].append(
                                f"optimize.bounds.{pside}.strategy.{strategy_name}"
                            )
                    for strategy_name in sorted(group_bounds):
                        if strategy_name not in handled_strategy_names:
                            report["manual_review_fields"].append(
                                f"optimize.bounds.{pside}.strategy.{strategy_name}"
                            )
    sort_optimize_bounds_in_place(target_bounds, strategy_kind=TRAILING_GRID_V7_KIND)


def _known_legacy_side_keys() -> set[str]:
    return set(STRATEGY_FIELD_MAP) | set(FLAT_BOT_KEY_TO_GROUP_PATH) | {
        "forager_score_weights",
        "forager_volatility_ema_span",
        "forager_volume_ema_span",
        "hsl_tier_ratio_yellow",
        "hsl_tier_ratio_orange",
        *LEGACY_SHARED_SIDE_ALIASES,
    }


def _report_source_strategy_subtree(
    source_side: dict,
    *,
    pside: str,
    report: dict,
) -> None:
    if "strategy" not in source_side:
        return
    strategy = source_side["strategy"]
    if strategy:
        report["manual_review_fields"].append(f"bot.{pside}.strategy")


def _has_v7_distinctive_strategy_fields(bot: dict) -> bool:
    for pside in ("long", "short"):
        source_side = bot.get(pside)
        if not isinstance(source_side, dict):
            continue
        if any(key in source_side for key in V7_DISTINCTIVE_STRATEGY_KEYS):
            return True
    return False


def _report_coin_override_leftovers(
    override: dict,
    *,
    coin: str,
    report: dict,
) -> None:
    supported_top = {"live", "override_config_path", "bot"}
    for key in sorted(override):
        if key not in supported_top:
            report["manual_review_fields"].append(f"coin_overrides.{coin}.{key}")

    bot = override.get("bot")
    if bot is None:
        return
    if not isinstance(bot, dict):
        report["manual_review_fields"].append(f"coin_overrides.{coin}.bot")
        return
    for key in sorted(bot):
        if key not in ("long", "short"):
            report["manual_review_fields"].append(f"coin_overrides.{coin}.bot.{key}")

    known_side_keys = _known_legacy_side_keys() | COIN_OVERRIDE_SIDE_PASSTHROUGH_KEYS
    for pside in ("long", "short"):
        side = bot.get(pside)
        if side is None:
            continue
        if not isinstance(side, dict):
            report["manual_review_fields"].append(f"coin_overrides.{coin}.bot.{pside}")
            continue
        for key in sorted(side):
            if key not in known_side_keys:
                report["manual_review_fields"].append(f"coin_overrides.{coin}.bot.{pside}.{key}")


def _migrate_coin_overrides(source: dict, target: dict, report: dict) -> None:
    coin_overrides = source.get("coin_overrides")
    if not isinstance(coin_overrides, dict):
        return
    allowed_live_keys = set(target.get("live", {}))
    result = {}
    for coin, override in coin_overrides.items():
        if not isinstance(override, dict):
            report["manual_review_fields"].append(f"coin_overrides.{coin}")
            continue
        migrated_override = {}
        if "live" in override:
            live_override = override.get("live")
            if isinstance(live_override, dict):
                migrated_live = {}
                for key, value in live_override.items():
                    path = f"coin_overrides.{coin}.live.{key}"
                    if key in allowed_live_keys:
                        migrated_live[key] = deepcopy(value)
                        report["moved_fields"].append(f"{path} -> {path}")
                    else:
                        report["manual_review_fields"].append(path)
                if migrated_live:
                    migrated_override["live"] = migrated_live
            else:
                report["manual_review_fields"].append(f"coin_overrides.{coin}.live")
        if "override_config_path" in override:
            migrated_override["override_config_path"] = deepcopy(override["override_config_path"])
        bot = override.get("bot")
        if isinstance(bot, dict):
            migrated_bot = {}
            for pside in ("long", "short"):
                side = bot.get(pside)
                if not isinstance(side, dict):
                    continue
                migrated_side = {}
                prefix = f"coin_overrides.{coin}.bot.{pside}"
                _move_shared_side_fields(
                    side,
                    migrated_side,
                    pside,
                    report,
                    source_prefix=prefix,
                    target_prefix=prefix,
                    allowed_flat_keys=COIN_OVERRIDE_SUPPORTED_SHARED_FLAT_KEYS,
                )
                _move_strategy_side_fields(
                    side,
                    migrated_side,
                    pside,
                    report,
                    source_prefix=prefix,
                    target_prefix=prefix,
                )
                for key in COIN_OVERRIDE_SIDE_PASSTHROUGH_KEYS:
                    if key in side:
                        migrated_side[key] = deepcopy(side[key])
                        report["moved_fields"].append(f"{prefix}.{key} -> {prefix}.{key}")
                if migrated_side:
                    migrated_bot[pside] = migrated_side
            if migrated_bot:
                migrated_override["bot"] = migrated_bot
        _report_coin_override_leftovers(override, coin=coin, report=report)
        if migrated_override:
            result[coin] = migrated_override
    if result:
        target["coin_overrides"] = result


def migrate_v7_trailing_grid_config(source: dict, *, source_path: str | None = None) -> tuple[dict, dict]:
    if not isinstance(source, dict):
        raise TypeError(f"source config must be a dict; got {type(source).__name__}")

    target = get_template_config()
    target["config_version"] = CONFIG_SCHEMA_VERSION
    target.setdefault("live", {})["strategy_kind"] = TRAILING_GRID_V7_KIND
    report = {
        "source_config_path": source_path,
        "source_version": source.get("config_version"),
        "destination_strategy_kind": TRAILING_GRID_V7_KIND,
        "moved_fields": [],
        "dropped_unsupported_fields": [],
        "manual_review_fields": [],
        "inserted_v8_defaults": [],
        "warnings": [],
    }
    _copy_supported_top_level_sections(source, target, report)
    _report_unknown_top_level_sections(source, report)
    target["live"]["strategy_kind"] = TRAILING_GRID_V7_KIND
    _prune_strategy_store(target)

    bot = source.get("bot")
    if not isinstance(bot, dict):
        raise ValueError("v7 config must contain a bot object")
    if not _has_v7_distinctive_strategy_fields(bot):
        raise ValueError(
            "input does not look like a supported v7 trailing-grid config: no v7-distinctive "
            "entry/close trailing-grid fields found under bot.long/bot.short"
        )

    moved_strategy = False
    for pside in ("long", "short"):
        source_side = bot.get(pside)
        if not isinstance(source_side, dict):
            report["manual_review_fields"].append(f"bot.{pside}")
            continue
        target_side = target["bot"][pside]
        _move_shared_side_fields(source_side, target_side, pside, report)
        moved_strategy = _move_strategy_side_fields(source_side, target_side, pside, report) or moved_strategy
        _report_source_strategy_subtree(source_side, pside=pside, report=report)
        known = _known_legacy_side_keys() | {"strategy"}
        for key in sorted(source_side):
            if key in BOT_GROUP_FIELD_MAP:
                report["manual_review_fields"].append(f"bot.{pside}.{key}")
            elif key not in known:
                report["manual_review_fields"].append(f"bot.{pside}.{key}")

    if not moved_strategy:
        raise ValueError(
            "input does not look like a supported v7 trailing-grid config: no legacy trailing-grid "
            "strategy fields found under bot.long/bot.short"
        )

    _move_bounds(source, target, report)
    _migrate_coin_overrides(source, target, report)
    _force_v7_absent_risk_defaults(source, target, report)
    _disable_enforcers_for_zero_v7_thresholds(source, target, report)
    _record_inserted_v8_defaults(source, target, report)
    _warn_if_v7_excess_would_be_clamped(target, report)
    return target, report


def migrate_v7_trailing_grid_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    allow_manual_review_output: bool = False,
) -> tuple[dict, dict]:
    from config.parse import load_raw_config
    import json

    input_path = Path(input_path)
    output_path = Path(output_path)
    source = load_raw_config(input_path)
    migrated, report = migrate_v7_trailing_grid_config(source, source_path=str(input_path))
    unresolved = migration_report_has_unresolved(report)
    report["manual_review_required"] = unresolved
    report["allow_manual_review_output"] = bool(allow_manual_review_output)
    report["output_config_path"] = str(output_path)
    if unresolved and not allow_manual_review_output:
        report["output_written"] = False
        report["status"] = "manual_review_required"
        return migrated, report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(migrated, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    report["output_written"] = True
    report["status"] = (
        "unsafe_manual_review_output_written"
        if unresolved
        else "ok"
    )
    return migrated, report
