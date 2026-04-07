import logging
from copy import deepcopy

from .bot import ensure_optimize_bounds_for_bot, format_bot_config
from .hydrate import (
    apply_non_live_adjustments,
    hydrate_missing_template_fields,
    preserve_coin_sources,
    reject_backtest_inherited_live_fields,
    seed_missing_compatibility_sections,
    sync_with_template,
)
from .coerce import normalize_validation_fields
from .migrations import apply_migrations, build_base_config_from_flavor, detect_flavor
from .scoring import normalize_scoring_config
from .schema import get_template_config
from .transform_log import ConfigTransformTracker, record_transform
from .access import require_config_dict
from .validate import validate_config


def normalize_config(
    config: dict,
    *,
    base_config_path: str = "",
    live_only: bool = False,
    verbose: bool = True,
    record_step: bool = True,
) -> dict:
    raw_snapshot = deepcopy(config["_raw"]) if "_raw" in config else None
    existing_log = config.get("_transform_log")
    if isinstance(existing_log, list):
        existing_log = deepcopy(existing_log)
    else:
        existing_log = []
    tracker = ConfigTransformTracker()
    optimize_suite_defined = (
        isinstance(config.get("optimize"), dict) and "suite" in config["optimize"]
    )
    raw_optimize_limits_present = (
        isinstance(config.get("optimize"), dict) and "limits" in config["optimize"]
    )
    raw_optimize_limits = deepcopy(config.get("optimize", {}).get("limits"))
    raw_optimize_snapshot = (
        deepcopy(config.get("optimize")) if isinstance(config.get("optimize"), dict) else {}
    )
    coin_sources_input = deepcopy(config.get("backtest", {}).get("coin_sources"))
    live_coin_sources_input = {}
    template = get_template_config()
    flavor = detect_flavor(config, template)
    result = build_base_config_from_flavor(config, template, flavor, verbose)
    if flavor == "nested_current" and isinstance(config.get("config"), dict):
        source_sections = set(config["config"])
    else:
        source_sections = set(config) if isinstance(config, dict) else set()
    for section in ("backtest", "bot", "coin_overrides", "live", "logging", "monitor", "optimize"):
        if section in result and section not in source_sections:
            tracker.add([section], result[section])
    for path in ("backtest", "bot", "live", "optimize"):
        require_config_dict(result, path)
    apply_migrations(result, verbose=verbose, tracker=tracker)
    for key in ("approved_coins", "ignored_coins"):
        if isinstance(result.get("live"), dict) and key in result["live"]:
            live_coin_sources_input[key] = deepcopy(result["live"][key])
    seed_missing_compatibility_sections(template, result, tracker=tracker)
    for path in ("bot.long", "bot.short", "optimize.bounds"):
        require_config_dict(result, path)

    result["bot"] = format_bot_config(
        result["bot"],
        live_cfg=result["live"],
        verbose=verbose,
        tracker=tracker,
    )
    ensure_optimize_bounds_for_bot(result, verbose=verbose, tracker=tracker)
    hydrate_missing_template_fields(template, result, verbose=verbose, tracker=tracker)
    reject_backtest_inherited_live_fields(result)
    sync_with_template(
        template,
        result,
        base_config_path,
        verbose=verbose,
        tracker=tracker,
    )
    normalize_validation_fields(result, raw_optimize=raw_optimize_snapshot)
    normalize_scoring_config(result, verbose=verbose, tracker=tracker)
    validate_config(
        result,
        raw_optimize=raw_optimize_snapshot,
        verbose=verbose,
        tracker=tracker,
    )

    if coin_sources_input is not None:
        result.setdefault("backtest", {})["coin_sources"] = coin_sources_input
    preserve_coin_sources(result, live_sources_input=live_coin_sources_input)

    if optimize_suite_defined:
        logging.warning(
            "Config contains optimize.suite, but suite configuration is now defined via "
            "backtest.scenarios. optimize.suite will be ignored and deleted; backtest.scenarios "
            "will be used. If you need different suite definitions, pass --suite-config with a "
            "file containing backtest.scenarios."
        )
        if isinstance(result.get("optimize"), dict) and "suite" in result["optimize"]:
            del result["optimize"]["suite"]

    if not live_only:
        apply_non_live_adjustments(
            result,
            verbose=verbose,
            tracker=tracker,
            raw_optimize_limits=raw_optimize_limits,
            raw_optimize_limits_present=raw_optimize_limits_present,
        )

    result["_transform_log"] = existing_log
    if raw_snapshot is not None and "_raw" not in result:
        result["_raw"] = deepcopy(raw_snapshot)

    if record_step:
        details = tracker.merge_details(
            {
                "live_only": live_only,
                "base_config_path": base_config_path,
                "flavor": flavor,
            }
        )
        record_transform(result, "normalize_config", details)
    return result
