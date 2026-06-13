from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from config.optimize_bounds import (
    BOT_BOUND_GROUP_BY_KEY,
    set_flat_optimize_bound,
    sort_optimize_bounds_in_place,
)
from config.schema import CONFIG_SCHEMA_VERSION, get_template_config
from config.shared_bot import BOT_GROUP_FIELD_MAP, FLAT_BOT_KEY_TO_GROUP_PATH


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

LEGACY_BOUND_ALIASES = {
    "min_markup": "close_grid_markup_start",
    "close_grid_min_markup": "close_grid_markup_end",
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

COIN_OVERRIDE_SIDE_PASSTHROUGH_KEYS = {
    "wallet_exposure_limit",
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


def _set_path(mapping: dict, path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = deepcopy(value)


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


def _move_shared_side_fields(
    source_side: dict,
    target_side: dict,
    pside: str,
    report: dict,
    *,
    source_prefix: str | None = None,
    target_prefix: str | None = None,
) -> None:
    source_prefix = source_prefix or f"bot.{pside}"
    target_prefix = target_prefix or f"bot.{pside}"
    written_targets: set[tuple[str, str]] = set()
    for flat_key, (group_name, local_key) in FLAT_BOT_KEY_TO_GROUP_PATH.items():
        if flat_key not in source_side:
            continue
        value = source_side[flat_key]
        if flat_key == "hsl_tier_ratios" and isinstance(value, dict):
            target_side.setdefault(group_name, {})[local_key] = deepcopy(value)
        else:
            target_side.setdefault(group_name, {})[local_key] = deepcopy(value)
        written_targets.add((group_name, local_key))
        report["moved_fields"].append(
            f"{source_prefix}.{flat_key} -> {target_prefix}.{group_name}.{local_key}"
        )

    for legacy_key, local_key in (
        ("hsl_tier_ratio_yellow", "tier_ratios.yellow"),
        ("hsl_tier_ratio_orange", "tier_ratios.orange"),
    ):
        if legacy_key not in source_side:
            continue
        _set_path(target_side.setdefault("hsl", {}), tuple(local_key.split(".")), source_side[legacy_key])
        written_targets.add(("hsl", local_key.split(".", 1)[0]))
        report["moved_fields"].append(
            f"{source_prefix}.{legacy_key} -> {target_prefix}.hsl.{local_key}"
        )

    if isinstance(source_side.get("forager_score_weights"), dict):
        target_side.setdefault("forager", {})["score_weights"] = deepcopy(
            source_side["forager_score_weights"]
        )
        written_targets.add(("forager", "score_weights"))
        report["moved_fields"].append(
            f"{source_prefix}.forager_score_weights -> {target_prefix}.forager.score_weights"
        )

    for legacy_key, local_key in (
        ("forager_volatility_ema_span", "volatility_ema_span_1m"),
        ("forager_volume_ema_span", "volume_ema_span_1m"),
    ):
        if legacy_key not in source_side:
            continue
        target_side.setdefault("forager", {})[local_key] = deepcopy(source_side[legacy_key])
        written_targets.add(("forager", local_key))
        report["moved_fields"].append(
            f"{source_prefix}.{legacy_key} -> {target_prefix}.forager.{local_key}"
        )

    seen_alias_targets: set[str] = set()
    for legacy_key, canonical_flat_key in LEGACY_SHARED_SIDE_ALIASES.items():
        if legacy_key not in source_side:
            continue
        group_name, local_key = FLAT_BOT_KEY_TO_GROUP_PATH[canonical_flat_key]
        target_group = target_side.setdefault(group_name, {})
        target_path = (group_name, local_key)
        if (
            canonical_flat_key not in source_side
            and canonical_flat_key not in seen_alias_targets
            and target_path not in written_targets
        ):
            target_group[local_key] = deepcopy(source_side[legacy_key])
            seen_alias_targets.add(canonical_flat_key)
            written_targets.add(target_path)
        report["moved_fields"].append(
            f"{source_prefix}.{legacy_key} -> {target_prefix}.{group_name}.{local_key}"
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
    for legacy_key, path in STRATEGY_FIELD_MAP.items():
        if legacy_key not in source_side:
            continue
        _set_path(strategy, path, source_side[legacy_key])
        report["moved_fields"].append(
            f"{source_prefix}.{legacy_key} -> {target_prefix}.strategy.{TRAILING_GRID_V7_KIND}.{'.'.join(path)}"
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
    return local in BOT_BOUND_GROUP_BY_KEY or local in STRATEGY_FIELD_MAP


def _move_bounds(source: dict, target: dict, report: dict) -> None:
    bounds = source.get("optimize", {}).get("bounds")
    if not isinstance(bounds, dict):
        return
    target_bounds = target.setdefault("optimize", {}).setdefault("bounds", {})
    if any(isinstance(key, str) and key.startswith(("long_", "short_")) for key in bounds):
        seen_canonical_keys: set[str] = set()
        for key, value in bounds.items():
            if not isinstance(key, str) or not key.startswith(("long_", "short_")):
                report["manual_review_fields"].append(f"optimize.bounds.{key}")
                continue
            canonical_key = _legacy_bound_key(key.split("_", 1)[0], key)
            if canonical_key is None:
                report["dropped_unsupported_fields"].append(f"optimize.bounds.{key}")
                continue
            if not _is_supported_flat_bound_key(canonical_key):
                report["manual_review_fields"].append(f"optimize.bounds.{key}")
                continue
            if canonical_key not in seen_canonical_keys:
                set_flat_optimize_bound(target_bounds, TRAILING_GRID_V7_KIND, canonical_key, value)
                seen_canonical_keys.add(canonical_key)
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
                    target_bounds.setdefault(pside, {})[group_name] = deepcopy(group_bounds)
                    report["moved_fields"].append(f"optimize.bounds.{pside}.{group_name}")
                    continue
                if isinstance(group_bounds, dict):
                    legacy_strategy = group_bounds.get("trailing_grid") or group_bounds.get(
                        TRAILING_GRID_V7_KIND
                    )
                    if isinstance(legacy_strategy, dict):
                        target_bounds[pside]["strategy"][TRAILING_GRID_V7_KIND] = deepcopy(
                            legacy_strategy
                        )
                        report["moved_fields"].append(
                            f"optimize.bounds.{pside}.strategy -> optimize.bounds.{pside}.strategy.{TRAILING_GRID_V7_KIND}"
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
        *COIN_OVERRIDE_SIDE_PASSTHROUGH_KEYS,
    }


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

    known_side_keys = _known_legacy_side_keys()
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
    result = {}
    for coin, override in coin_overrides.items():
        if not isinstance(override, dict):
            report["manual_review_fields"].append(f"coin_overrides.{coin}")
            continue
        migrated_override = {}
        if isinstance(override.get("live"), dict):
            migrated_override["live"] = deepcopy(override["live"])
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
    }
    _copy_supported_top_level_sections(source, target, report)
    target["live"]["strategy_kind"] = TRAILING_GRID_V7_KIND
    _prune_strategy_store(target)

    bot = source.get("bot")
    if not isinstance(bot, dict):
        raise ValueError("v7 config must contain a bot object")

    moved_strategy = False
    for pside in ("long", "short"):
        source_side = bot.get(pside)
        if not isinstance(source_side, dict):
            report["manual_review_fields"].append(f"bot.{pside}")
            continue
        target_side = target["bot"][pside]
        _move_shared_side_fields(source_side, target_side, pside, report)
        moved_strategy = _move_strategy_side_fields(source_side, target_side, pside, report) or moved_strategy
        known = _known_legacy_side_keys() | {"strategy"}
        for key in sorted(source_side):
            if key not in known and key not in BOT_GROUP_FIELD_MAP:
                report["manual_review_fields"].append(f"bot.{pside}.{key}")

    if not moved_strategy:
        raise ValueError(
            "input does not look like a supported v7 trailing-grid config: no legacy trailing-grid "
            "strategy fields found under bot.long/bot.short"
        )

    _move_bounds(source, target, report)
    _migrate_coin_overrides(source, target, report)
    return target, report


def migrate_v7_trailing_grid_file(input_path: str | Path, output_path: str | Path) -> tuple[dict, dict]:
    from config.parse import load_raw_config
    import json

    input_path = Path(input_path)
    output_path = Path(output_path)
    source = load_raw_config(input_path)
    migrated, report = migrate_v7_trailing_grid_config(source, source_path=str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(migrated, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    return migrated, report
