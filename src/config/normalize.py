from copy import deepcopy

from .bot import (
    ensure_optimize_bounds_for_bot,
    format_bot_config,
)
from .hydrate import (
    apply_non_live_adjustments,
    hydrate_missing_template_fields,
    preserve_coin_sources,
    reject_backtest_inherited_live_fields,
    seed_missing_default_sections,
    sync_with_template,
)
from .coerce import normalize_validation_fields
from .gpu import resolve_gpu_screening
from .prefilter import resolve_prefilter
from .optimize_bounds import prune_inactive_optimize_strategy_bounds
from .scoring import normalize_scoring_config
from .schema import get_template_config
from .hsl import normalization_template, normalize_hsl, require_current_hsl_schema
from .strategy import (
    prune_inactive_strategy_subtrees,
    reject_legacy_flat_strategy_fields,
    sync_canonical_strategy_config,
)
from .transform_log import ConfigTransformTracker, record_transform
from .access import require_config_dict
from .validate import validate_config
from .universe import get_side_universe_sources


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
    from .authored import validate_authored_config
    from .migrations.v9 import require_v9_runtime_config

    require_v9_runtime_config(config)
    validate_authored_config(config, get_template_config())
    flavor = "canonical"
    source_payload = config
    require_current_hsl_schema(source_payload, base_config_path=base_config_path)
    raw_optimize_limits_present = (
        isinstance(source_payload.get("optimize"), dict)
        and "limits" in source_payload["optimize"]
    )
    raw_optimize_limits = deepcopy(source_payload.get("optimize", {}).get("limits"))
    raw_optimize_snapshot = (
        deepcopy(source_payload.get("optimize"))
        if isinstance(source_payload.get("optimize"), dict)
        else {}
    )
    coin_sources_input = deepcopy(
        source_payload.get("backtest", {}).get("coin_sources")
    )
    live_coin_sources_input = {}
    template = normalization_template(get_template_config(), source_payload)
    result = deepcopy(config)
    source_sections = set(config)
    for section, defaults in template.items():
        if section not in result:
            result[section] = deepcopy(defaults)
    for section in (
        "backtest",
        "bot",
        "coin_overrides",
        "live",
        "logging",
        "monitor",
        "optimize",
    ):
        if section in result and section not in source_sections:
            tracker.add([section], result[section])
    for path in ("backtest", "bot", "live", "optimize"):
        require_config_dict(result, path)
    reject_legacy_flat_strategy_fields(result)
    result["optimize"]["prefilter"] = resolve_prefilter(result["optimize"].get("prefilter"))
    gpu = result["optimize"].get("gpu")
    if isinstance(gpu, dict) and "screening" in gpu:
        gpu["screening"] = resolve_gpu_screening(gpu["screening"])
    for key in ("approved_coins", "ignored_coins"):
        value = get_side_universe_sources(result, key)
        if value is not None:
            live_coin_sources_input[key] = deepcopy(value)
    seed_missing_default_sections(template, result, tracker=tracker)
    for path in ("bot.long", "bot.short", "optimize.bounds"):
        require_config_dict(result, path)
    reject_legacy_flat_strategy_fields(result)
    normalize_hsl(result, template, verbose=verbose)
    result["bot"] = format_bot_config(
        result["bot"],
        live_cfg=result["live"],
        verbose=verbose,
        tracker=tracker,
    )
    sync_canonical_strategy_config(result, tracker=tracker)
    ensure_optimize_bounds_for_bot(
        result,
        verbose=verbose,
        tracker=tracker,
        source_bounds=raw_optimize_snapshot.get("bounds", {}),
    )
    # All missing numeric dimensions were fixed from effective bot values above;
    # later default hydration must not reopen the profile's recommended ranges.
    template["optimize"]["bounds"] = deepcopy(result["optimize"]["bounds"])
    hydrate_missing_template_fields(template, result, verbose=verbose, tracker=tracker)
    reject_backtest_inherited_live_fields(result)
    sync_with_template(
        template,
        result,
        base_config_path,
        verbose=verbose,
        tracker=tracker,
    )
    sync_canonical_strategy_config(result, tracker=tracker)
    prune_inactive_strategy_subtrees(result, tracker=tracker)
    prune_inactive_optimize_strategy_bounds(result, tracker=tracker)
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

    if not live_only:
        apply_non_live_adjustments(
            result,
            verbose=verbose,
            tracker=tracker,
            raw_optimize_limits=raw_optimize_limits,
            raw_optimize_limits_present=raw_optimize_limits_present,
        )

    normalize_hsl(result, template, verbose=verbose)
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
