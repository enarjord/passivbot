import argparse
import json
import logging
import re
from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Tuple, List, Union, Optional, Iterable

from config.access import (
    get_optional_config_value,
    get_optional_live_value,
    require_config_dict,
    require_config_value,
    require_live_value,
)
from config.bot import (
    ensure_bot_defaults as staged_ensure_bot_defaults,
    ensure_optimize_bounds_for_bot as staged_ensure_optimize_bounds_for_bot,
    format_bot_config as staged_format_bot_config,
    normalize_position_counts as staged_normalize_position_counts,
)
from config.load import load_prepared_config as staged_load_prepared_config
from config.coerce import (
    HSL_SIGNAL_MODES,
    MONITOR_BOOL_KEYS,
    PYMOO_ALGORITHMS,
    PYMOO_REF_DIR_METHODS,
    normalize_hsl_signal_mode,
)
from config.gpu import parse_screening_scenarios
from config.hydrate import (
    PARTIALLY_OPEN_CONFIG_PATHS,
    apply_non_live_adjustments as staged_apply_non_live_adjustments,
    hydrate_missing_template_fields as staged_hydrate_missing_template_fields,
    sync_with_template as staged_sync_with_template,
)
from config.limits import (
    _resolve_optimize_limits_for_load,
    normalize_limit_entries,
    parse_limit_cli_entries,
    parse_limit_cli_entry,
    parse_limits_string,
)
from config.log_output import log_config_message
from config.metrics import CURRENCY_METRICS, SHARED_METRICS
from config.risk_memory import (
    normalize_memory_ema_span_minutes,
    normalize_risk_bootstrap_max_lookback_minutes,
)
from config.normalize import normalize_config
from config.optimize_bounds import prune_inactive_optimize_strategy_bounds
from config.overrides import (
    apply_allowed_modifications as staged_apply_allowed_modifications,
    get_allowed_modifications as staged_get_allowed_modifications,
    load_override_config as staged_load_override_config,
    nested_update as staged_nested_update,
    parse_old_coin_flags as staged_parse_old_coin_flags,
    parse_overrides as staged_parse_overrides,
    set_nested_value as staged_set_nested_value,
    set_nested_value_safe as staged_set_nested_value_safe,
)
from config.parse import load_raw_config
from config.project import project_config
from config.runtime_compile import compile_runtime_config
from config.schema import (
    CONFIG_SCHEMA_VERSION,
    get_template_config as get_schema_template_config,
)
from config.strategy import prune_inactive_strategy_subtrees
from config.tree_ops import (
    add_missing_keys_recursively,
    remove_unused_keys_recursively,
)
from config.transform_log import ConfigTransformTracker, record_transform
from config.logging_summary import emit_transform_summary
from config.migrations import (
    apply_backward_compatibility_renames as apply_migration_renames,
    migrate_btc_collateral_settings as migrate_btc_collateral_settings_impl,
    migrate_config_version as migrate_config_version_impl,
    migrate_empty_means_all_approved as migrate_empty_means_all_approved_impl,
    migrate_suite_to_scenarios as migrate_suite_to_scenarios_impl,
    rename_config_keys as rename_migration_config_keys,
)
from pure_funcs import sort_dict_keys, str2bool
from utils import dump_json_streamlined, normalize_coins_source, symbol_to_coin


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    log_config_message(verbose, level, message, *args)


def _canonical_log_json(value: Any) -> str:
    return json.dumps(
        sort_dict_keys(deepcopy(value)), sort_keys=True, separators=(",", ":")
    )


def _format_optimize_limits_change_for_log(
    old_value: Any, new_value: Any
) -> Optional[str]:
    if not isinstance(old_value, list) or not isinstance(new_value, list):
        return None
    if not all(isinstance(x, dict) for x in old_value) or not all(
        isinstance(x, dict) for x in new_value
    ):
        return None

    old_counter = Counter(_canonical_log_json(entry) for entry in old_value)
    new_counter = Counter(_canonical_log_json(entry) for entry in new_value)

    added: list[Any] = []
    removed: list[Any] = []

    for payload, count in sorted((new_counter - old_counter).items()):
        added.extend(json.loads(payload) for _ in range(count))
    for payload, count in sorted((old_counter - new_counter).items()):
        removed.extend(json.loads(payload) for _ in range(count))

    if not added and not removed:
        return "changed optimize.limits normalized formatting only"

    parts: list[str] = ["changed optimize.limits"]
    if added:
        noun = "entry" if len(added) == 1 else "entries"
        parts.append(f"added {len(added)} {noun}: {added}")
    if removed:
        noun = "entry" if len(removed) == 1 else "entries"
        parts.append(f"removed {len(removed)} {noun}: {removed}")
    return " | ".join(parts)


_APPROVED_COINS_LOG_SAMPLE_SIZE = 3
_APPROVED_COINS_LOG_SYMBOL_MAX_LENGTH = 8


def _format_approved_coins_symbol_for_log(value: Any) -> str:
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    symbol = (
        value[:_APPROVED_COINS_LOG_SYMBOL_MAX_LENGTH]
        .replace("\n", " ")
        .replace("\r", " ")
    )
    if len(value) > _APPROVED_COINS_LOG_SYMBOL_MAX_LENGTH:
        return f"{symbol}..."
    return symbol


def _format_approved_coins_collection_for_log(value: list[Any]) -> str:
    count = len(value)
    if not count:
        return "0[]"
    sample = [
        _format_approved_coins_symbol_for_log(symbol)
        for symbol in value[:_APPROVED_COINS_LOG_SAMPLE_SIZE]
    ]
    if count > _APPROVED_COINS_LOG_SAMPLE_SIZE:
        sample.append(f"+{count - _APPROVED_COINS_LOG_SAMPLE_SIZE}")
    return f"{count}[{','.join(sample)}]"


def _format_approved_coins_list_for_log(value: list[Any]) -> str:
    return f"list:{_format_approved_coins_collection_for_log(value)}"


def _format_approved_coins_shape_for_log(value: Any) -> str:
    if isinstance(value, list):
        return _format_approved_coins_list_for_log(value)
    if isinstance(value, dict):
        return f"dict:{len(value)}"
    if isinstance(value, str):
        return f"str:{len(value)}"
    if isinstance(value, (tuple, set, frozenset)):
        return f"{type(value).__name__}:{len(value)}"
    if value is None:
        return "none"
    return f"type={type(value).__name__}"


def _format_approved_coins_change_for_log(old_value: Any, new_value: Any, *, path: str) -> str:
    def format_value(value: Any) -> str:
        if (
            isinstance(value, dict)
            and set(value) == {"long", "short"}
            and isinstance(value["long"], list)
            and isinstance(value["short"], list)
        ):
            return (
                "sides("
                f"long:{_format_approved_coins_collection_for_log(value['long'])},"
                f"short:{_format_approved_coins_collection_for_log(value['short'])}"
                ")"
            )
        return _format_approved_coins_shape_for_log(value)

    return (
        f"changed {path} "
        f"{format_value(old_value)} -> {format_value(new_value)}"
    )


def _format_config_change_message(
    full_path: str, old_value: Any, new_value: Any
) -> tuple[str, tuple[Any, ...]]:
    if full_path in {f"bot.{side}.universe.approved_coins" for side in ("long", "short")}:
        return _format_approved_coins_change_for_log(old_value, new_value, path=full_path), ()
    if full_path == "optimize.limits":
        custom = _format_optimize_limits_change_for_log(old_value, new_value)
        if custom is not None:
            return custom, ()
    return "changed %s %s -> %s", (full_path, old_value, new_value)


Path = Tuple[str, ...]  # ("bot", "long", "entry_grid_spacing_pct")
FIELD_RUNTIME_RULES = {
    "backtest.offline": {
        "owner": "backtest",
        "consumed_by": {"backtest", "optimize"},
        "cli_exposed_on": {"backtest", "optimize"},
        "help_group": {"backtest": "Backtest Runtime", "optimize": "Backtest Runtime"},
    },
    "approved_coins": {
        "owner": "bot",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
    },
    "ignored_coins": {
        "owner": "bot",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
    },
    "bot.global.universe.approved_coins": {"owner": "bot", "consumed_by": {"live", "backtest", "optimize"}, "cli_exposed_on": {"live", "backtest", "optimize"}, "help_group": {"backtest": "Coin Selection", "optimize": "Coin Selection"}},
    "bot.long.universe.minimum_coin_age_days": {
        "owner": "bot",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
    },
    "bot.short.universe.minimum_coin_age_days": {
        "owner": "bot",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
    },
    "bot.global.forager.score_hysteresis_margin": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
    },
    "live.recovery.exchange_symbol_unavailable_cooldown_hours": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "live.execution.order_replacement_churn_gate_activation_count": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "live.execution.order_replacement_churn_gate_market_distance_fraction": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "live.execution.order_replacement_churn_gate_stability_minutes": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "live.execution.order_replacement_churn_gate_window_minutes": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "live.execution.limit_order_create_max_market_distance_fraction": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "bot.global.hedge_mode": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "bot.global.execution.market_orders_allowed": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "bot.global.execution.market_order_near_touch_fraction": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "live.fills.fallback_fee_fraction": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "live.fills.max_abs_fee_fraction": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "live.fills.fee_conversion_max_age_milliseconds": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "bot.global.loss_gate.max_drawdown_ratio": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "bot.global.warmup.max_minutes": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Date Range",
            "optimize": "Backtest Runtime",
        },
    },
    "bot.global.unstuck.memory_ema_span_minutes": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
    "live.startup.risk_bootstrap_max_lookback_minutes": {
        "owner": "live",
        "consumed_by": {"live"},
        "cli_exposed_on": {"live"},
    },
    "bot.global.hsl.scope": {
        "owner": "live",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
    },
}

OPTIMIZE_FIXED_BOT_RUNTIME_CLI_ARGS = {
    "bot.long.hsl.enabled": {
        "visible": ["--bot.long.hsl.enabled"],
        "hidden": ["--bot.long.hsl_enabled", "--bot_long_hsl_enabled"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"optimize"},
        "help": "Override bot.long.hsl.enabled for this optimize run.",
    },
    "bot.short.hsl.enabled": {
        "visible": ["--bot.short.hsl.enabled"],
        "hidden": ["--bot.short.hsl_enabled", "--bot_short_hsl_enabled"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"optimize"},
        "help": "Override bot.short.hsl.enabled for this optimize run.",
    },
    "bot.long.hsl.panic_close_order_type": {
        "visible": ["--bot.long.hsl.panic_close_order_type"],
        "hidden": [
            "--bot.long.hsl_panic_close_order_type",
            "--bot_long_hsl_panic_close_order_type",
        ],
        "type": str,
        "metavar": "VALUE",
        "commands": {"optimize"},
        "choices": ["limit", "market"],
        "help": "Override bot.long.hsl.panic_close_order_type for this optimize run.",
    },
    "bot.short.hsl.panic_close_order_type": {
        "visible": ["--bot.short.hsl.panic_close_order_type"],
        "hidden": [
            "--bot.short.hsl_panic_close_order_type",
            "--bot_short_hsl_panic_close_order_type",
        ],
        "type": str,
        "metavar": "VALUE",
        "commands": {"optimize"},
        "choices": ["limit", "market"],
        "help": "Override bot.short.hsl.panic_close_order_type for this optimize run.",
    },

}


for _pside in ("long", "short"):
    _path = f"bot.{_pside}.hsl.scale_budget_with_excess_allowance"
    OPTIMIZE_FIXED_BOT_RUNTIME_CLI_ARGS[_path] = {
        "visible": [f"--{_path}"],
        "hidden": [],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"optimize"},
        "help": "Scale the coin HSL budget by bounded excess headroom for this optimize run.",
    }


def get_field_runtime_rule(full_name: str) -> dict:
    return FIELD_RUNTIME_RULES.get(full_name, {})


def field_cli_exposed_on(full_name: str, command: Optional[str]) -> bool:
    if command is None:
        return True
    rule = get_field_runtime_rule(full_name)
    return command in rule.get("cli_exposed_on", set())


def field_cli_help_group(full_name: str, command: Optional[str]) -> Optional[str]:
    if command is None:
        return None
    rule = get_field_runtime_rule(full_name)
    help_group = rule.get("help_group", {})
    if not isinstance(help_group, dict):
        return None
    return help_group.get(command)


def load_hjson_config(config_path: str, *, log_errors: bool = True) -> dict:
    return load_raw_config(config_path, log_errors=log_errors)


def load_config(filepath: str, live_only=False, verbose=True) -> dict:
    try:
        config = staged_load_prepared_config(
            filepath,
            live_only=live_only,
            verbose=verbose,
            target="canonical",
            runtime=None,
            log_info=False,
        )
        existing_log = config.get("_transform_log", [])
        config["_transform_log"] = []
        record_transform(config, "load_config", {"path": filepath})
        config["_transform_log"].extend(existing_log)
        return config
    except Exception:
        logging.exception("failed to load config %s", filepath)
        raise


def dump_config(config: dict, filepath: str, *, clean: bool = False):
    from config.universe import anchored_universe_source_view
    config_copy = anchored_universe_source_view(config)
    if clean:
        config_copy = clean_config(config_copy)
    sorted_config = sort_dict_keys(config_copy)
    try:
        with open(filepath, "w", encoding="utf-8") as fp:
            dump_json_streamlined(sorted_config, fp, sort_keys=False)
            fp.write("\n")
    except Exception:
        logging.exception("failed to dump config to %s", filepath)
        raise


def expand_PB_mode(mode: str) -> str:
    if mode.lower() in ["gs", "graceful_stop", "graceful-stop"]:
        return "graceful_stop"
    elif mode.lower() in ["m", "manual"]:
        return "manual"
    elif mode.lower() in ["n", "normal"]:
        return "normal"
    elif mode.lower() in ["p", "panic"]:
        return "panic"
    elif mode.lower() in ["t", "tp", "tp_only", "tp-only"]:
        return "tp_only"
    else:
        raise Exception(f"unknown passivbot mode {mode}")


def apply_allowed_modifications(
    src, modifications, allowed_overrides, return_full=True
):
    return staged_apply_allowed_modifications(
        src, modifications, allowed_overrides, return_full=return_full
    )


def get_allowed_modifications():
    return staged_get_allowed_modifications()


def set_nested_value(d: dict, p: list, v: object):
    staged_set_nested_value(d, p, v)


def set_nested_value_safe(d: dict, p: list, v: object, create_missing=False):
    return staged_set_nested_value_safe(d, p, v, create_missing=create_missing)


def nested_update(base_dict, update_dict):
    return staged_nested_update(base_dict, update_dict)


def parse_overrides(config, verbose=True):
    return staged_parse_overrides(
        config,
        verbose=verbose,
        override_loader=load_override_config,
        symbol_normalizer=symbol_to_coin,
    )


def load_override_config(config, coin):
    return staged_load_override_config(config, coin)


def parse_old_coin_flags(config) -> dict:
    return staged_parse_old_coin_flags(config)


def _apply_backward_compatibility_renames(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    apply_migration_renames(result, verbose=verbose, tracker=tracker)


def _migrate_suite_to_scenarios(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    migrate_suite_to_scenarios_impl(result, verbose=verbose, tracker=tracker)


def _migrate_btc_collateral_settings(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    migrate_btc_collateral_settings_impl(result, verbose=verbose, tracker=tracker)


def _migrate_config_version(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    migrate_config_version_impl(result, verbose=verbose, tracker=tracker)


def _migrate_empty_means_all_approved(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    migrate_empty_means_all_approved_impl(result, verbose=verbose, tracker=tracker)


def _ensure_bot_defaults_and_bounds(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    _ensure_bot_defaults(result, verbose=verbose, tracker=tracker)
    _ensure_optimize_bounds_for_bot(result, verbose=verbose, tracker=tracker)


def _ensure_bot_defaults(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    staged_ensure_bot_defaults(result, verbose=verbose, tracker=tracker)


def _ensure_optimize_bounds_for_bot(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    staged_ensure_optimize_bounds_for_bot(result, verbose=verbose, tracker=tracker)


def format_bot_config(
    bot_cfg: dict,
    *,
    live_cfg: Optional[dict] = None,
    verbose: bool = True,
    warn_deprecations: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
) -> dict:
    return staged_format_bot_config(
        bot_cfg,
        live_cfg=live_cfg,
        verbose=verbose,
        warn_deprecations=warn_deprecations,
        tracker=tracker,
    )


def _rename_config_keys(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    rename_migration_config_keys(result, verbose=verbose, tracker=tracker)


def _sync_with_template(
    template: dict,
    result: dict,
    base_config_path: str,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
) -> None:
    staged_sync_with_template(
        template,
        result,
        base_config_path,
        verbose=verbose,
        tracker=tracker,
    )


def _hydrate_missing_template_fields(
    template: dict,
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
) -> None:
    staged_hydrate_missing_template_fields(
        template, result, verbose=verbose, tracker=tracker
    )


def _normalize_position_counts(
    result: dict, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    staged_normalize_position_counts(result, tracker=tracker)


def _apply_non_live_adjustments(
    result: dict,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
    raw_optimize_limits: Any = None,
    raw_optimize_limits_present: Optional[bool] = None,
) -> None:
    staged_apply_non_live_adjustments(
        result,
        verbose=verbose,
        tracker=tracker,
        raw_optimize_limits=raw_optimize_limits,
        raw_optimize_limits_present=raw_optimize_limits_present,
    )


def format_config(
    config: dict, verbose=True, live_only=False, base_config_path: str = ""
) -> dict:
    result = normalize_config(
        config,
        base_config_path=base_config_path,
        live_only=live_only,
        verbose=verbose,
        record_step=True,
    )
    latest_details = {}
    existing_log = result.get("_transform_log")
    if isinstance(existing_log, list) and existing_log:
        latest_details = deepcopy(existing_log[-1].get("details", {}))
    record_transform(
        result,
        "format_config",
        latest_details
        or {
            "live_only": live_only,
            "base_config_path": base_config_path,
        },
    )
    emit_transform_summary(result, step="format_config", verbose=verbose)
    return result


def _clean_dynamic_node(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, sub_value in value.items():
            if str(key).startswith("_"):
                continue
            cleaned[key] = _clean_dynamic_node(sub_value)
        return cleaned
    if isinstance(value, list):
        return [_clean_dynamic_node(item) for item in value]
    return deepcopy(value)


def _clean_with_template(template_node, source_node, path: Path = ()):
    if isinstance(template_node, dict):
        source_dict = source_node if isinstance(source_node, dict) else {}
        if path in PARTIALLY_OPEN_CONFIG_PATHS:
            if not isinstance(source_node, dict):
                return _clean_with_template(template_node, None, ())
            cleaned = {}
            for key, value in source_dict.items():
                if key in template_node:
                    cleaned[key] = _clean_with_template(
                        template_node[key], value, path + (key,)
                    )
                else:
                    cleaned[key] = _clean_dynamic_node(value)
            if path == ("backtest", "reducer") and "default" not in cleaned:
                cleaned["default"] = _clean_with_template(
                    template_node["default"],
                    None,
                    path + ("default",),
                )
            return cleaned
        if not template_node:
            return _clean_dynamic_node(source_dict)
        result = {}
        for key, tmpl_value in template_node.items():
            result[key] = _clean_with_template(
                tmpl_value, source_dict.get(key), path + (key,)
            )
        return result
    if isinstance(template_node, list):
        if isinstance(source_node, list):
            return [_clean_dynamic_node(item) for item in source_node]
        return deepcopy(template_node)
    if source_node is None:
        return deepcopy(template_node)
    return deepcopy(source_node)


def clean_config(config: dict) -> dict:
    """
    Return a sanitized config aligned with the template structure, stripped of helper keys,
    with dictionaries sorted recursively.
    """
    from config.hsl import FIELDS, normalization_template
    # Source validation and template preparation are read-only. The projection
    # below copies retained values, so a full input copy would be redundant.
    from config.universe import universe_source_view
    source = universe_source_view({key: value for key, value in (config or {}).items()
                                   if not isinstance(key, str) or not key.startswith("_")})
    from config.migrations.v9 import require_v9_runtime_config, reject_retired_v9_fields

    if "config_version" in source:
        require_v9_runtime_config(source)
    else:
        reject_retired_v9_fields(source)
    template = normalization_template(get_template_config(), source)
    portfolio = source.get("bot", {}).get("global", {}).get("hsl", {})
    if not isinstance(portfolio, dict):
        raise TypeError("bot.global.hsl must be a mapping")
    # Retain explicit portfolio policy without hydrating missing authorization.
    template["bot"]["global"]["hsl"].update(
        {key: None for key in FIELDS if key in portfolio}
    )
    portfolio_bounds = source.get("optimize", {}).get("bounds", {}).get("global", {}).get("hsl")
    if portfolio_bounds is not None:
        if not isinstance(portfolio_bounds, dict):
            raise TypeError("optimize.bounds.global.hsl must be a mapping")
        template["optimize"]["bounds"].setdefault("global", {})["hsl"] = {
            key: None for key in FIELDS if key in portfolio_bounds
        }
    from config.optimize_bounds import (
        hydrate_adaptive_optimize_bounds,
        preserve_optional_adaptive_bounds,
    )

    preserve_optional_adaptive_bounds(template, source)
    cleaned = _clean_with_template(template, source)
    cleaned["bot"]["global"]["universe"]["approved_coins"] = deepcopy(
        source.get("bot", {}).get("global", {}).get("universe", {}).get("approved_coins", [])
    )
    # Derive fixed weight bounds from the same normalized weights the loader
    # will consume, while preserving the cleaner's authored bot values.
    bound_config = {**cleaned, "bot": deepcopy(cleaned["bot"])}
    from config.bot import normalize_bot_forager_config

    normalize_bot_forager_config(bound_config, verbose=False)
    hydrate_adaptive_optimize_bounds(
        bound_config, source_bounds=source.get("optimize", {}).get("bounds", {})
    )
    prune_inactive_strategy_subtrees(cleaned)
    prune_inactive_optimize_strategy_bounds(cleaned)
    return sort_dict_keys(cleaned)


def strip_config_metadata(config: dict, *, keys: Iterable[str] | None = None) -> dict:
    """
    Return a deep-copied config with the provided metadata keys removed recursively.
    Defaults to removing `_raw`, `_raw_effective`, and `_transform_log`.
    """

    removal = set(
        keys or ("_raw", "_raw_effective", "_transform_log", "_coins_sources")
    )

    def _strip(node):
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items() if k not in removal}
        if isinstance(node, list):
            return [_strip(item) for item in node]
        return deepcopy(node)

    from config.universe import universe_source_view
    return _strip(universe_source_view(config))


def sanitize_prepared_config_for_dump(
    config: dict, *, extra_keys: Iterable[str] | None = None
) -> dict:
    """
    Return a prepared config stripped of runtime/metadata payload so the dumped artifact remains
    clean and directly reusable.
    """

    removal = ["_raw", "_raw_effective", "_transform_log", "_coins_sources", "analysis"]
    if extra_keys is not None:
        removal.extend(extra_keys)
    from config.universe import anchored_universe_source_view
    stripped = strip_config_metadata(anchored_universe_source_view(config), keys=tuple(removal))
    return clean_config(stripped)


def _limits_structurally_equal(
    raw_limits: Any, normalized_limits: List[Dict[str, Any]]
) -> bool:
    if not isinstance(raw_limits, list) or len(raw_limits) != len(normalized_limits):
        return False

    def _scalar_equal(left: Any, right: Any) -> bool:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)) <= 1e-12
        return left == right

    def _value_equal(left: Any, right: Any) -> bool:
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                _scalar_equal(l_item, r_item) for l_item, r_item in zip(left, right)
            )
        return _scalar_equal(left, right)

    for raw_entry, normalized_entry in zip(raw_limits, normalized_limits):
        if not isinstance(raw_entry, dict) or set(raw_entry) != set(normalized_entry):
            return False
        for key, normalized_value in normalized_entry.items():
            if not _value_equal(raw_entry.get(key), normalized_value):
                return False
    return True


def comma_separated_values_float(x):
    return [float(z) for z in x.split(",")]


def comma_separated_values(x):
    # Preserve JSON/HJSON-like strings (used for approved/ignored coin dicts)
    if isinstance(x, str):
        raw = x.strip()
        if raw and raw[0] in "[{" and raw[-1] in "]}":
            return [x]
    return [item.strip() for item in x.split(",")]


def optional_float(x):
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return float(x)


def merge_negative_cli_values(argv):
    """Allow comma-separated values that begin with '-' to be parsed as option values."""
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            out.extend(argv[i:])
            break
        if token.startswith("-") and "=" not in token and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and "," in nxt:
                out.append(f"{token}={nxt}")
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def create_acronym(full_name, acronyms=set()):
    i = 1
    while True:
        i += 1
        if i > 100:
            raise Exception(f"too many acronym duplicates for {full_name}")
        shortened_name = full_name
        for k in [
            "backtest.",
            "live.",
            "optimize.bounds.",
            "optimize.limits.",
            "optimize.",
            "bot.",
        ]:
            if shortened_name.startswith(k):
                shortened_name = shortened_name.replace(k, "")
                break

        # Split on both '_' and '.' using regex
        splitted = re.split(r"[._]+", shortened_name)
        acronym = "".join(word[0] for word in splitted if word)  # skip any empty splits

        if acronym not in acronyms:
            break
        acronym = acronym + str(i)
        if acronym not in acronyms:
            break
    return acronym


# Hard-coded CLI shortcuts for backwards compatibility and cleaner default help.
# Format:
#   config_key -> {
#       "visible": ["--preferred-name", "-x"],
#       "hidden": ["--legacy_name", "--legacy_name_with_dots"],
#       "command_aliases": {"optimize": ["-x"]},
#       "commands": {"live", "backtest", "optimize"},
#       "group": {"live": "Coin Selection", ...},
#       "type": type_converter,
#       "metavar": "CSV|INT|...",
#       "help": "Human-facing help text",
#   }
RESERVED_CLI_ARGS = {
    "approved_coins": {
        "visible": ["--symbols", "-s"],
        "hidden": [],
        "type": comma_separated_values,
        "metavar": "CSV_OR_PATH",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": (
            "Approved coin sources for both sides. Use CSV like BTC,ETH,XRP, the literal 'all', a path to a JSON "
            "coin list file, or a JSON/HJSON per-side object like "
            '{"long":["BTC"],"short":"all"}. Use coin tickers, not exchange symbols.'
        ),
    },
    "ignored_coins": {
        "visible": ["--ignored-coins", "-ic"],
        "hidden": [],
        "type": comma_separated_values,
        "metavar": "CSV_OR_PATH",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Ignored coin sources for both sides. CSV, a JSON/HJSON per-side object, or a JSON coin list file.",
    },
    "bot.long.universe.minimum_coin_age_days": {
        "visible": ["--minimum-coin-age-days-long"],
        "hidden": ["--bot_long_universe_minimum_coin_age_days"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Minimum coin age in days for initial entries on this side; zero bypasses, held management is retained.",
    },
    "bot.short.universe.minimum_coin_age_days": {
        "visible": ["--minimum-coin-age-days-short"],
        "hidden": ["--bot_short_universe_minimum_coin_age_days"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Minimum coin age in days for initial entries on this side; zero bypasses, held management is retained.",
    },
    "bot.global.forager.score_hysteresis_margin": {
        "visible": ["--forager-score-hysteresis-margin"],
        "hidden": ["--bot_global_forager_score_hysteresis_margin"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Forager incumbent score tolerance. Keeps an already-selected flat forager coin when a challenger score is within this fractional normalized-score gap.",
    },
    "live.recovery.exchange_symbol_unavailable_cooldown_hours": {
        "visible": ["--exchange-symbol-unavailable-cooldown-hours"],
        "hidden": [
            "--live.recovery.exchange_symbol_unavailable_cooldown_hours",
            "--live_recovery_exchange_symbol_unavailable_cooldown_hours",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": (
            "Hours to suppress new entries for a symbol after a connector proves "
            "that the venue has temporarily disabled API trading for it. The "
            "RAM-only cooldown resets on bot restart; 0 disables this suppression."
        ),
    },
    "live.execution.order_replacement_churn_gate_activation_count": {
        "visible": ["--order-replacement-churn-gate-activation-count"],
        "hidden": [
            "--live.execution.order_replacement_churn_gate_activation_count",
            "--live_execution_order_replacement_churn_gate_activation_count",
        ],
        "type": int,
        "metavar": "INT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Account-wide rolling create-attempt count that activates economy deferral for far unstable orders. Set to 0 to disable only that deferral; reconciliation safety remains active.",
    },
    "live.execution.order_replacement_churn_gate_market_distance_fraction": {
        "visible": ["--order-replacement-churn-gate-market-distance-fraction"],
        "hidden": [
            "--live.execution.order_replacement_churn_gate_market_distance_fraction",
            "--live_execution_order_replacement_churn_gate_market_distance_fraction",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Final market-distance threshold that always admits a limit order despite churn allowance exhaustion.",
    },
    "live.execution.order_replacement_churn_gate_stability_minutes": {
        "visible": ["--order-replacement-churn-gate-stability-minutes"],
        "hidden": [
            "--live.execution.order_replacement_churn_gate_stability_minutes",
            "--live_execution_order_replacement_churn_gate_stability_minutes",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Minimum sustained drift duration; the same tight-match duration clears older evidence.",
    },
    "live.execution.order_replacement_churn_gate_window_minutes": {
        "visible": ["--order-replacement-churn-gate-window-minutes"],
        "hidden": [
            "--live.execution.order_replacement_churn_gate_window_minutes",
            "--live_execution_order_replacement_churn_gate_window_minutes",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Window for per-symbol ideal-order evidence and account-wide create-attempt accounting.",
    },
    "live.execution.limit_order_create_max_market_distance_fraction": {
        "visible": ["--limit-order-create-max-market-distance-fraction"],
        "hidden": [
            "--live.execution.limit_order_create_max_market_distance_fraction",
            "--live_execution_limit_order_create_max_market_distance_fraction",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": (
            "Live pre-create guard for limit orders. Set to 0 to disable. "
            "With the default 0.8, buys below market*0.2 and sells above "
            "market*1.8 are skipped until price action moves closer."
        ),
    },
    "bot.global.hedge_mode": {
        "visible": ["--hedge-mode", "-hm"],
        "hidden": ["--bot_global_hedge_mode"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": (
            "Enable or disable hedge mode. If the exchange does not support simultaneous "
            "long and short on the same coin, the bot will use hedge_mode=false."
        ),
    },
    "live.account.leverage": {
        "visible": ["--leverage", "-lev"],
        "hidden": ["--live.account.leverage", "--live_account_leverage"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Set leverage for this run.",
    },
    "bot.global.execution.market_orders_allowed": {
        "visible": ["--market-orders-allowed", "-moa"],
        "hidden": ["--bot_global_execution_market_orders_allowed"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": (
            "Allow Rust to promote crossing/near-touch orders to live market orders. "
            "Disabled by default; live exchange slippage controls apply."
        ),
    },
    "bot.global.execution.market_order_near_touch_fraction": {
        "visible": ["--market-order-near-touch-fraction", "-montf"],
        "hidden": ["--bot_global_execution_market_order_near_touch_fraction"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": (
            "Fractional distance threshold for promoting near-touch orders to market "
            "when market orders are allowed; this is not a live slippage cap."
        ),
    },
    "bot.global.loss_gate.max_drawdown_ratio": {
        "visible": ["--loss-gate-max-drawdown-ratio"],
        "hidden": ["--bot_global_loss_gate_max_drawdown_ratio"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": "Limit non-panic close losses using independent loss-gate drawdown memory; >=1 disables the gate.",
    },
    "live.fills.fallback_fee_fraction": {
        "visible": ["--fallback-fee-fraction"],
        "hidden": ["--live.fills.fallback_fee_fraction", "--live_fills_fallback_fee_fraction"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": "Fallback fill fee fraction of absolute quote notional when reported fee data cannot be converted to quote currency.",
    },
    "live.fills.max_abs_fee_fraction": {
        "visible": ["--max-abs-fee-fraction"],
        "hidden": [
            "--live.fills.max_abs_fee_fraction",
            "--live_fills_max_abs_fee_fraction",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": "Absolute fee/notional sanity threshold; outliers are replaced with live.fills.fallback_fee_fraction.",
    },
    "live.fills.fee_conversion_max_age_milliseconds": {
        "visible": ["--fee-conversion-max-age-milliseconds"],
        "hidden": [
            "--live.fills.fee_conversion_max_age_milliseconds",
            "--live_fills_fee_conversion_max_age_milliseconds",
        ],
        "type": int,
        "metavar": "INT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Maximum age difference allowed when using ticker data to convert non-quote fee tokens.",
    },
    "bot.global.unstuck.memory_ema_span_minutes": {
        "visible": ["--unstuck-memory-ema-span-minutes"],
        "hidden": ["--bot_global_unstuck_memory_ema_span_minutes"],
        "type": normalize_memory_ema_span_minutes,
        "metavar": "FLOAT",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": "Global unstuck realized-memory EMA span in minutes.",
    },
    "live.startup.risk_bootstrap_max_lookback_minutes": {
        "visible": ["--risk-bootstrap-max-lookback-minutes"],
        "hidden": [
            "--live.startup.risk_bootstrap_max_lookback_minutes",
            "--live_startup_risk_bootstrap_max_lookback_minutes",
        ],
        "type": normalize_risk_bootstrap_max_lookback_minutes,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Live reconstruction cap in minutes (maximum 90 days); requested horizon is five slow spans.",
    },
    "bot.global.hsl.scope": {
        "visible": ["--hsl-scope"],
        "hidden": ["--bot_global_hsl_scope"],
        "type": normalize_hsl_signal_mode,
        "metavar": "MODE",
        "choices": tuple(HSL_SIGNAL_MODES),
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Behavior",
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": (
            "HSL signal mode. portfolio uses one account-level strategy signal for both "
            "sides; position_side tracks long/short independently; coin tracks each coin+side "
            "slot and panic-closes only the affected slot."
        ),
    },
    "live.execution.time_in_force": {
        "visible": ["--time-in-force", "-tif"],
        "hidden": ["--live.execution.time_in_force", "--live_execution_time_in_force"],
        "type": str,
        "metavar": "VALUE",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Time-in-force policy for live orders.",
    },
    "live.account.user": {
        "visible": ["--user", "-u"],
        "hidden": ["--live.account.user", "--live_account_user"],
        "type": str,
        "metavar": "VALUE",
        "commands": {"live"},
        "group": {"live": "Runtime"},
        "help": "Account name from api-keys.json to use for this live run.",
    },
    "backtest.exchanges": {
        "visible": ["--exchanges", "-e"],
        "hidden": ["--backtest.exchanges", "--backtest_exchanges"],
        "type": comma_separated_values,
        "metavar": "CSV",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Backtest exchanges to use, for example bybit or binance,bybit.",
    },
    "backtest.start_date": {
        "visible": ["--start-date", "-sd"],
        "hidden": ["--backtest.start_date", "--backtest_start_date"],
        "type": str,
        "metavar": "DATE",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Date Range",
            "optimize": "Date Range",
        },
        "help": "Backtest start date. Examples: 2025, 2025-01, 2025-01-15.",
    },
    "backtest.end_date": {
        "visible": ["--end-date", "-ed"],
        "hidden": ["--backtest.end_date", "--backtest_end_date"],
        "type": str,
        "metavar": "DATE",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Date Range",
            "optimize": "Date Range",
        },
        "help": 'Backtest end date. Use "-ed now" for the latest available candles.',
    },
    "backtest.candle_interval_minutes": {
        "visible": ["--candle-interval-minutes", "-cim"],
        "hidden": [
            "--backtest.candle_interval_minutes",
            "--backtest_candle_interval_minutes",
        ],
        "type": int,
        "metavar": "INT",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Backtest Runtime",
            "optimize": "Date Range",
        },
        "help": "Backtest candle interval in minutes.",
    },
    "backtest.offline": {
        "visible": ["--offline"],
        "hidden": ["--backtest.offline", "--backtest_offline"],
        "type": str2bool,
        "metavar": "BOOL",
        "commands": {"backtest", "optimize"},
        "help": "Use local simulation data only; never refresh metadata or download candles.",
    },
    "backtest.hlcvs_data_dir": {
        "visible": ["--hlcvs-data-dir"],
        "hidden": ["--backtest.hlcvs_data_dir", "--backtest_hlcvs_data_dir"],
        "type": str,
        "metavar": "PATH",
        "commands": {"backtest"},
        "group": {"backtest": "Backtest Runtime"},
        "help": "Use a specific final HLCV dataset directory instead of resolving by config hash.",
    },
    "backtest.hlcvs_data_override_mode": {
        "visible": ["--hlcvs-data-override-mode"],
        "hidden": [
            "--backtest.hlcvs_data_override_mode",
            "--backtest_hlcvs_data_override_mode",
        ],
        "type": str,
        "metavar": "MODE",
        "commands": {"backtest"},
        "group": {"backtest": "Backtest Runtime"},
        "choices": ("intersection", "dataset"),
        "help": (
            "How --hlcvs-data-dir chooses coins/range. intersection keeps the current "
            "config clipped to the verified dataset; dataset adopts the dataset's "
            "effective coins and timestamp window for exact artifact replay."
        ),
    },
    "backtest.maker_fee_override": {
        "visible": ["--maker-fee-override"],
        "hidden": ["--backtest.maker_fee_override", "--backtest_maker_fee_override"],
        "type": optional_float,
        "metavar": "FLOAT",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": "Override all backtest maker fees. Leave unset to use exchange-derived per-coin maker fees.",
    },
    "backtest.taker_fee_override": {
        "visible": ["--taker-fee-override"],
        "hidden": ["--backtest.taker_fee_override", "--backtest_taker_fee_override"],
        "type": optional_float,
        "metavar": "FLOAT",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Backtest Runtime",
            "optimize": "Backtest Runtime",
        },
        "help": "Override all backtest taker fees. Leave unset to use exchange-derived per-coin taker fees.",
    },
    "backtest.starting_balance": {
        "visible": ["--starting-balance", "-sb"],
        "hidden": ["--backtest.starting_balance", "--backtest_starting_balance"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"backtest"},
        "group": {"backtest": "Backtest Runtime"},
        "help": "Starting balance for the backtest.",
    },
    "backtest.reducer.default": {
        "visible": ["--reducer-default"],
        "hidden": [
            "--backtest.reducer.default",
            "--backtest_reducer_default",
            "--aggregate-default",
            "--backtest.aggregate.default",
            "--backtest_aggregate_default",
            "--backtest.stat.default",
            "--backtest_stat_default",
            "--backtest.scenario_stat.default",
            "--backtest_scenario_stat_default",
        ],
        "type": str,
        "metavar": "MODE",
        "commands": {"backtest", "optimize"},
        "group": {"backtest": "Suite", "optimize": "Suite"},
        "help": (
            "Suite-only default reducer for scenario metrics. Allowed modes: "
            "mean, min, max, std, median. Metric-specific backtest.reducer entries "
            "override this default."
        ),
    },
    "optimize.iters": {
        "visible": ["--iters", "-i"],
        "hidden": ["--optimize.iters", "--optimize_iters"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer iteration budget.",
    },
    "optimize.n_cpus": {
        "visible": ["--cpus", "-c"],
        "hidden": ["--optimize.n_cpus", "--optimize_n_cpus"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer worker count.",
    },
    "optimize.scoring": {
        "visible": ["--scoring", "-os"],
        "hidden": ["--optimize.scoring", "--optimize_scoring"],
        "type": comma_separated_values,
        "metavar": "CSV",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer scoring metrics, for example adg_pnl,loss_profit_ratio.",
    },
    "optimize.objective_scenario": {
        "visible": ["--objective-scenario"],
        "hidden": ["--optimize.objective_scenario", "--optimize_objective_scenario"],
        "type": str,
        "metavar": "LABEL",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": (
            "Default suite scenario for scoring objectives. Per-objective scenario selectors "
            "may override it; optimize.limits remain suite-aggregated."
        ),
    },
    "optimize.population_size": {
        "visible": ["--population-size", "-ps"],
        "hidden": ["--optimize.population_size", "--optimize_population_size"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer population size.",
    },
    "optimize.pareto_max_size": {
        "visible": ["--pareto-max-size", "-pms"],
        "hidden": ["--optimize.pareto_max_size", "--optimize_pareto_max_size"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Maximum persisted Pareto set size.",
    },
    "optimize.backend": {
        "visible": ["--backend", "-ob"],
        "hidden": ["--optimize.backend", "--optimize_backend", "--optimizer-backend"],
        "type": str,
        "metavar": "BACKEND",
        "choices": ("pymoo", "gpu"),
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer backend: pymoo or gpu (authoritative NVIDIA/CUDA or experimental Apple Metal simulations).",
    },
    "optimize.limits": {
        "visible": ["--limits"],
        "hidden": ["--optimize.limits", "--optimize_limits"],
        "type": str,
        "metavar": "JSON_OR_HJSON",
        "commands": {"optimize"},
        "group": {"optimize": "Optimize Common"},
        "help": "Replace optimize.limits for this run with a JSON/HJSON list of limit objects.",
    },
    "optimize.pymoo.shared.mutation_prob": {
        "visible": ["--optimize.pymoo.shared.mutation_prob"],
        "hidden": [
            "--optimize_pymoo_shared_mutation_prob",
            "--optimize.pymoo.shared.mutation_prob_var",
            "--optimize_pymoo_shared_mutation_prob_var",
            "-psmpv",
        ],
        "type": str,
        "metavar": "VALUE",
        "commands": {"optimize"},
        "help": "Per-individual mutation probability, or auto for 1 / n_params.",
    },
}

# Keep these convenience aliases stable across config grouping changes.
for _pside in ("long", "short"):
    for _param, _acronym, _help in (
        ("total_wallet_exposure_limit", "twel", "Total wallet exposure limit"),
        ("max_positions", "np", "Target number of concurrent position slots"),
    ):
        _key = f"bot.{_pside}.risk.{_param}"
        RESERVED_CLI_ARGS[_key] = {
            "visible": [f"--{_key}", f"-{_pside[0]}{_acronym}"],
            "hidden": [f"--{_key.replace('.', '_')}", f"-{_pside[0]}r{_acronym}"],
            "type": float,
            "metavar": "FLOAT",
            "commands": {"live", "backtest"},
            "group": {"live": "Behavior", "backtest": "Backtest Runtime"},
            "help": f"{_help} for the {_pside} side.",
        }

# Optimizer shortcuts set search bounds rather than live/backtest bot values.
for _pside in ("long", "short"):
    _key = f"optimize.bounds.{_pside}.risk.total_wallet_exposure_limit"
    RESERVED_CLI_ARGS[_key] = {
        "visible": [f"--{_key}"],
        "command_aliases": {"optimize": [f"-{_pside[0]}twel"]},
        "hidden": [f"--{_key.replace('.', '_')}", f"-{_pside[0]}rtwel"],
        "type": comma_separated_values_float,
        "metavar": "VALUE_OR_RANGE",
        "commands": {"optimize"},
        "group": {"optimize": "Optimize Bounds"},
        "help": f"Wallet exposure bounds for the {_pside} side: VALUE or LOW,HIGH[,STEP].",
    }

for _path in ("bot.global.unstuck.memory_ema_span_minutes", "bot.global.loss_gate.max_drawdown_ratio", "bot.global.hsl.scope"):
    FIELD_RUNTIME_RULES[_path]["owner"] = "bot"
FIELD_RUNTIME_RULES["bot.global.loss_gate.memory_ema_span_minutes"] = deepcopy(
    FIELD_RUNTIME_RULES["bot.global.unstuck.memory_ema_span_minutes"]
)
RESERVED_CLI_ARGS["bot.global.loss_gate.memory_ema_span_minutes"] = {
    **deepcopy(RESERVED_CLI_ARGS["bot.global.unstuck.memory_ema_span_minutes"]),
    "visible": ["--loss-gate-memory-ema-span-minutes"],
    "hidden": ["--bot_global_loss_gate_memory_ema_span_minutes"],
    "help": "Global loss-gate realized-memory EMA span in minutes.",
}

for _side in ("long", "short"):
    _path = f"bot.{_side}.universe.filter_by_min_effective_cost"
    RESERVED_CLI_ARGS[_path] = {
        "visible": [f"--{_path}"], "hidden": [], "type": str2bool, "metavar": "Y/N",
        "commands": {"live", "backtest", "optimize"},
        "group": dict.fromkeys(("live", "backtest", "optimize"), "Coin Selection"),
        "help": f"Filter {_side} flat initial entries whose configured size is below the exchange minimum; backtest null inherits this flag.",
    }

RESERVED_CLI_ARGS.update(OPTIMIZE_FIXED_BOT_RUNTIME_CLI_ARGS)
RESERVED_CLI_ARGS["bot.global.universe.approved_coins"] = {
    **deepcopy(RESERVED_CLI_ARGS["approved_coins"]),
    "visible": ["--bot.global.universe.approved_coins"], "hidden": [],
    "help": "Shared approval source; explicit side sources keep their replacement authority.",
}


# Shared fixed policy remains configurable on every consuming command. The
# optimizer projects these settings separately from its numeric search genes.
for _path, _group in (
    ("bot.global.strategy_kind", "Behavior"),
    ("bot.global.warmup.span_multiplier", "Backtest Runtime"),
    ("bot.global.warmup.max_minutes", "Backtest Runtime"),
    ("bot.long.universe.filter_by_min_effective_cost", "Coin Selection"),
    ("bot.short.universe.filter_by_min_effective_cost", "Coin Selection"),
):
    FIELD_RUNTIME_RULES[_path] = {
        "owner": "bot",
        "consumed_by": {"live", "backtest", "optimize"},
        "cli_exposed_on": {"live", "backtest", "optimize"},
        "help_group": {
            "live": "Runtime" if _group == "Backtest Runtime" else _group,
            "backtest": "Backtest Runtime" if _group == "Behavior" else _group,
            "optimize": "Backtest Runtime" if _group == "Behavior" else _group,
        },
    }
for _pside in ("long", "short"):
    for _field in ("approved_coins", "ignored_coins"):
        _path = f"bot.{_pside}.universe.{_field}"
        FIELD_RUNTIME_RULES[_path] = deepcopy(FIELD_RUNTIME_RULES[_field])
        RESERVED_CLI_ARGS[_path] = {
            "visible": [f"--{_path}"], "hidden": [],
            "type": comma_separated_values, "metavar": "CSV_OR_PATH",
            "commands": {"live", "backtest", "optimize"},
            "group": {command: "Coin Selection" for command in ("live", "backtest", "optimize")},
            "help": f"{_field.replace('_', ' ').capitalize()} source for the {_pside} side only.",
        }
    for _path, _group in ((f"bot.{_pside}.forager.always_include", "Coin Selection"),):
        FIELD_RUNTIME_RULES[_path] = {
            "owner": "bot", "consumed_by": {"live", "backtest", "optimize"},
            "cli_exposed_on": {"live", "backtest", "optimize"},
            "help_group": {command: _group for command in ("live", "backtest", "optimize")},
        }
for _path, _rule in FIELD_RUNTIME_RULES.items():
    if _path.startswith("bot."):
        _rule["owner"] = "bot"
for _path, _spec in RESERVED_CLI_ARGS.items():
    if _path.startswith("bot.") and f"--{_path}" not in _spec["visible"]:
        _spec["visible"].append(f"--{_path}")


def _merge_runtime_rules_into_reserved_cli_args():
    for full_name, rule in FIELD_RUNTIME_RULES.items():
        spec = RESERVED_CLI_ARGS.get(full_name)
        if spec is None:
            continue

        cli_exposed_on = rule.get("cli_exposed_on", set())
        if cli_exposed_on:
            spec["commands"] = set(spec.get("commands", set())) | set(cli_exposed_on)

        help_group = rule.get("help_group", {})
        if isinstance(help_group, dict) and help_group:
            merged_group = dict(spec.get("group", {}))
            merged_group.update(help_group)
            spec["group"] = merged_group


_merge_runtime_rules_into_reserved_cli_args()

CLI_HELP_GROUPS = {
    "live": [
        "Coin Selection",
        "Behavior",
        "Runtime",
        "Logging",
        "Advanced Overrides",
    ],
    "backtest": [
        "Coin Selection",
        "Date Range",
        "Backtest Runtime",
        "Suite",
        "Output / Analysis",
        "Logging",
        "Advanced Overrides",
    ],
    "optimize": [
        "Coin Selection",
        "Date Range",
        "Optimizer",
        "Suite",
        "Logging",
        "Backtest Runtime",
        "Optimize Common",
        "Optimize Bounds",
        "Optimize Pymoo",
        "Advanced Overrides",
    ],
}


def _register_argument(container, visible_names, hidden_names, **kwargs):
    container.add_argument(*visible_names, **kwargs)
    hidden_kwargs = dict(kwargs)
    hidden_kwargs["help"] = argparse.SUPPRESS
    for alias in hidden_names:
        if alias in container._option_string_actions:
            continue
        container.add_argument(alias, **hidden_kwargs)


def _argument_metavar(type_, full_name: str, value):
    if type_ is comma_separated_values:
        return "CSV"
    if type_ is comma_separated_values_float:
        return "MIN,MAX[,STEP]"
    if type_ is str2bool:
        return "Y/N"
    if type_ is optional_float:
        return "FLOAT"
    if type_ is int:
        return "INT"
    if type_ is float:
        return "FLOAT"
    if type_ is str and full_name.endswith("_date"):
        return "DATE"
    if type_ is str and full_name.endswith("_dir"):
        return "PATH"
    if type_ is str and value is None:
        return "VALUE"
    return "VALUE"


CLI_HELP_OVERRIDES = {
    "optimize.enable_overrides": (
        "Optimizer convenience overrides. couple_unstuck_ema_spans derives each coin and side's "
        "unstuck horizons from its effective strategy, removing redundant unstuck span genes. "
        "Saved candidates retain explicit spans. Default: no overrides."
    ),
    "backtest.scenarios": (
        "Suite scenario definitions. Use --scenarios to select labels; use "
        "--suite-config for complex scenario files. Scenario entries support "
        "label, start_date, end_date, coins, ignored_coins, exchanges, "
        "coin_sources, and overrides."
    ),
    "backtest.balance_sample_divider": (
        "Minutes per saved balance/equity sample. 1 keeps per-minute series; "
        "higher values thin balance_and_equity.csv.gz and related plots."
    ),
    "backtest.btc_collateral_cap": (
        "Target and ceiling share of equity held as BTC collateral. 0 keeps "
        "USD-only; 1 targets fully BTC collateral; values >1 allow leveraged "
        "BTC collateral."
    ),
    "backtest.btc_collateral_ltv_cap": (
        "Optional USD-debt/equity cap while topping up BTC collateral. Leave "
        "unset for no LTV cap."
    ),
    "backtest.compress_cache": (
        "Compress generated backtest cache artifacts to save disk. Set n/false "
        "for faster reloads with larger files."
    ),
    "backtest.dynamic_wel_by_tradability": (
        "Backtest-only WEL denominator mode. y grows the denominator with the "
        "max tradable coin count seen so far; n uses fixed live-style "
        "max_positions."
    ),
    "backtest.organillo_csv": "Optional daily UTC date/coin 0/1 eligibility CSV; null disables Organillo.",
    "backtest.organillo_sha256": "Pinned Organillo input hash; leave null for a new input.",
    "backtest.filter_by_min_effective_cost": (
        "Skip coins whose projected initial entry is below the exchange "
        "effective min cost. y improves live parity; n allows oversized "
        "min-cost entries."
    ),
    "backtest.gap_tolerance_ohlcvs_minutes": (
        "Largest internal candle gap, in minutes, that preparation may "
        "tolerate or repair before excluding the affected tradable window."
    ),
    "backtest.ohlcv_source_dir": (
        "Read an explicit offline candle input, without converting the current store. Expected layout: "
        "<dir>/<exchange>/1m/<coin_or_symbol>/YYYY-MM-DD.npz or .npy."
    ),
    'bot.global.warmup.max_minutes': (
        "Hard cap on calculated EMA/history warmup minutes for backtests and "
        "live. null leaves the cap unbounded; otherwise use a positive number."
    ),
    "backtest.base_dir": "Directory where standalone backtest results are written.",
    "backtest.volume_normalization": (
        "Normalize combined-exchange volume with robust complete-day, cross-coin "
        "estimates. Disable to preserve each selected exchange's native volume."
    ),
    "backtest.equity_floor_ratio": (
        "Early-stop equity floor as a fraction of starting balance. Must "
        "satisfy 0 <= x < 1; 0.05 stops once equity is <= 5 percent of start."
    ),
    "backtest.limit_order_fill_buffer_fraction": (
        "Backtest-only required crossing beyond a limit price as part-per-one "
        "(0.0001 = 0.01%%). Strict crossing; fills retain the limit price. Default 0.0."
    ),
    "backtest.market_order_slippage_fraction": (
        "Backtest-only simulated market-order slippage as part-per-one. Applies "
        "to market-promoted orders and market HSL panic closes; not a live "
        "slippage cap."
    ),
    "backtest.suite_enabled": (
        "Enable suite mode from config. CLI --suite y/n overrides it for one run."
    ),
    "backtest.visible_metrics": (
        "Terminal metric visibility config. null uses optimize scoring/limits; "
        "[] shows all; a list adds named metrics. Full analysis is still saved."
    ),
    "config_version": (
        f"Source config schema version ({CONFIG_SCHEMA_VERSION} for final canonical V9). "
        "Cannot be overridden at runtime; use explicit migration tools."
    ),
}

for _pside in ("long", "short"):
    CLI_HELP_OVERRIDES.update(
        {
            f"bot.{_pside}.hsl.scale_budget_with_excess_allowance": (
                "Scale coin HSL balance budget by effective bounded excess allowance. "
                "Default false; allows coin overrides in coin HSL scope."
            ),
            f"bot.{_pside}.hsl.enabled": f"Enable HSL for the {_pside} side.",
            f"bot.{_pside}.hsl.drawdown_ratio_threshold": (
                f"RED drawdown trigger for {_pside} HSL, as part-per-one."
            ),
            f"bot.{_pside}.hsl.drawdown_ratio_ema_span_minutes": (
                f"EMA span, in minutes, for the {_pside} HSL drawdown signal."
            ),
            f"bot.{_pside}.hsl.cooldown_after_red_minutes": (
                f"Minutes to wait after {_pside} HSL RED is flattened before "
                "restart is allowed."
            ),
            f"bot.{_pside}.hsl.panic_close_order_type": (
                "Allowed values: limit or market. market uses "
                "backtest.market_order_slippage_fraction and taker fees in backtests."
            ),
            f"bot.{_pside}.risk.total_exposure_enforcer.enabled": (
                f"Enable {_pside} TWEL auto-reduce repair for already-over-target "
                "same-side exposure."
            ),
            f"bot.{_pside}.risk.total_exposure_enforcer.policy": (
                "Allowed values: reduce_overweight or reduce_portfolio. Controls "
                f"which managed {_pside} positions TWEL auto-reduce may trim."
            ),
            f"bot.{_pside}.risk.total_exposure_enforcer.threshold_ratio": (
                f"Fraction of {_pside} TWEL used by entry gating and TWEL "
                "auto-reduce repair."
            ),
            f"bot.{_pside}.risk.we_excess_allowance_fraction": (
                f"Per-symbol allowance above the configured {_pside} WEL before "
                "position exposure trimming."
            ),
            f"bot.{_pside}.risk.max_positions": (
                f"Target number of concurrent {_pside} position slots."
            ),
            f"bot.{_pside}.risk.total_wallet_exposure_limit": (
                f"Total wallet exposure limit for the {_pside} side."
            ),
        }
    )


def _argument_help_text(full_name: str, appendix: str) -> str:
    override = CLI_HELP_OVERRIDES.get(full_name)
    if override is not None:
        return override
    base = f"Override {full_name}."
    if appendix:
        return f"{base} {appendix}".strip()
    return base


def _classify_live_argument(full_name: str, help_all: bool) -> Optional[str]:
    shared_group = field_cli_help_group(full_name, "live")
    if shared_group is not None:
        return shared_group
    coin_selection = {
        "approved_coins",
        'bot.global.forager.score_hysteresis_margin',
        "ignored_coins",
        'bot.long.universe.minimum_coin_age_days',
        'bot.short.universe.minimum_coin_age_days',
    }
    behavior = {
        "bot.long.universe.filter_by_min_effective_cost",
        "bot.short.universe.filter_by_min_effective_cost",
        "live.fills.fee_conversion_max_age_milliseconds",
        "live.fills.fallback_fee_fraction",
        "live.fills.max_abs_fee_fraction",
        "live.modes.long",
        "live.modes.short",
        "bot.global.hedge_mode",
        "live.recovery.exchange_symbol_unavailable_cooldown_hours",
        "live.account.leverage",
        "bot.global.execution.market_orders_allowed",
        "bot.global.loss_gate.max_drawdown_ratio",
        "live.execution.limit_order_create_max_market_distance_fraction",
        "live.execution.order_match_tolerance_fraction",
        "live.execution.order_replacement_churn_gate_activation_count",
        "live.execution.order_replacement_churn_gate_market_distance_fraction",
        "live.execution.order_replacement_churn_gate_stability_minutes",
        "live.execution.order_replacement_churn_gate_window_minutes",
    }
    runtime = {
        "live.execution.execution_delay_seconds",
        "live.api.max_concurrent_api_requests",
        "live.recovery.max_n_restarts_per_day",
        "live.candles.max_ohlcv_fetches_per_minute",
        "live.api.recv_window_milliseconds",
        "live.execution.time_in_force",
        "live.account.user",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in behavior:
        return "Behavior"
    if full_name in runtime:
        return "Runtime"
    return "Advanced Overrides" if help_all else None


def _classify_backtest_argument(full_name: str, help_all: bool) -> Optional[str]:
    shared_group = field_cli_help_group(full_name, "backtest")
    if shared_group is not None:
        return shared_group
    coin_selection = {
        "backtest.exchanges",
    }
    date_range = {
        "backtest.end_date",
        "backtest.start_date",
    }
    runtime = {
        "backtest.reducer.default",
        "backtest.balance_sample_divider",
        "backtest.btc_collateral_cap",
        "backtest.btc_collateral_ltv_cap",
        "backtest.candle_interval_minutes",
        "backtest.compress_cache",
        "backtest.dynamic_wel_by_tradability",
        "backtest.filter_by_min_effective_cost",
        "backtest.gap_tolerance_ohlcvs_minutes",
        "backtest.maker_fee_override",
        "backtest.ohlcv_source_dir",
        "backtest.starting_balance",
        "backtest.taker_fee_override",
    }
    output = {
        "backtest.base_dir",
        "backtest.volume_normalization",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in date_range:
        return "Date Range"
    if full_name in runtime:
        return "Backtest Runtime"
    if full_name in output:
        return "Output / Analysis"
    if full_name.startswith("backtest.scenarios"):
        return "Suite"
    return "Advanced Overrides" if help_all else None


def _classify_optimize_argument(full_name: str, help_all: bool) -> Optional[str]:
    shared_group = field_cli_help_group(full_name, "optimize")
    if shared_group is not None:
        return shared_group
    coin_selection = {
        "backtest.exchanges",
    }
    date_range = {
        "backtest.end_date",
        "backtest.start_date",
        "backtest.candle_interval_minutes",
    }
    optimizer = {
        "optimize.iters",
        "optimize.backend",
        "optimize.n_cpus",
        "optimize.objective_scenario",
        "optimize.pareto_max_size",
        "optimize.population_size",
        "optimize.scoring",
    }
    backtest_runtime = {
        "backtest.reducer.default",
        "backtest.balance_sample_divider",
        "backtest.btc_collateral_cap",
        "backtest.btc_collateral_ltv_cap",
        "backtest.compress_cache",
        "backtest.dynamic_wel_by_tradability",
        "backtest.filter_by_min_effective_cost",
        "backtest.gap_tolerance_ohlcvs_minutes",
        "backtest.maker_fee_override",
        "backtest.ohlcv_source_dir",
        "backtest.starting_balance",
        "backtest.taker_fee_override",
        "backtest.volume_normalization",
    }
    optimize_common = {
        "optimize.compress_results_file",
        "optimize.enable_overrides",
        "optimize.fixed_params",
        "optimize.fixed_runtime_overrides",
        "optimize.limits",
        "optimize.max_pending_starting_evals_per_cpu",
        "optimize.round_to_n_significant_digits",
        "optimize.write_all_results",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in date_range:
        return "Date Range"
    if full_name in optimizer:
        return "Optimizer"
    if full_name in backtest_runtime:
        return "Backtest Runtime"
    if full_name.startswith("backtest.scenarios"):
        return "Suite"
    if full_name.startswith("optimize.bounds."):
        return "Optimize Bounds" if help_all else None
    if full_name.startswith("optimize.pymoo."):
        return "Optimize Pymoo" if help_all else None
    if full_name in optimize_common:
        return "Optimize Common" if help_all else None
    return "Advanced Overrides" if help_all else None


def classify_config_argument(
    full_name: str, command: Optional[str], help_all: bool
) -> Optional[str]:
    if command == "live":
        return _classify_live_argument(full_name, help_all)
    if command == "backtest":
        return _classify_backtest_argument(full_name, help_all)
    if command == "optimize":
        return _classify_optimize_argument(full_name, help_all)
    return None


def project_template_config_for_cli(config: dict, command: Optional[str]) -> dict:
    result = deepcopy(config)
    omitted = object()

    def fixed_policy(node, prefix):
        if field_cli_exposed_on(prefix, command):
            return deepcopy(node)
        if isinstance(node, dict):
            retained = {
                key: fixed_policy(value, f"{prefix}.{key}")
                for key, value in node.items()
            }
            return {
                key: value for key, value in retained.items() if value is not omitted
            } or omitted
        return omitted

    if command == "backtest":
        result.pop("optimize", None)
        live_cfg = result.get("live")
        if isinstance(live_cfg, dict):
            projected = fixed_policy(live_cfg, "live")
            result["live"] = {} if projected is omitted else projected
        return result
    if command == "optimize":
        projected = fixed_policy(result.get("bot", {}), "bot")
        result["bot"] = {} if projected is omitted else projected
        live_cfg = result.get("live")
        if isinstance(live_cfg, dict):
            projected = fixed_policy(live_cfg, "live")
            result["live"] = {} if projected is omitted else projected
        return result
    return result


def add_reserved_arguments(
    parser,
    *,
    command: Optional[str] = None,
    help_all: bool = False,
    group_map=None,
) -> Tuple[set, set]:
    """Add hard-coded CLI arguments for backwards compatibility.

    Returns the set of reserved acronyms and config keys that should be
    skipped by add_arguments_recursively().
    """
    reserved_acronyms = set()
    reserved_keys = set()

    for config_key, spec in RESERVED_CLI_ARGS.items():
        commands = spec.get("commands")
        if command is not None and commands is not None and command not in commands:
            continue
        visible_group = (
            spec.get("group", {}).get(command) if command is not None else None
        )
        container = (
            group_map.get(visible_group, parser)
            if group_map and visible_group
            else parser
        )

        help_text = (
            spec["help"]
            if help_all or visible_group is not None or command is None
            else argparse.SUPPRESS
        )
        if spec.get("action") == "store_true":
            register_kwargs = dict(
                action="store_true",
                dest=config_key,
                required=False,
                default=spec.get("default"),
                help=help_text,
            )
        else:
            register_kwargs = dict(
                type=spec["type"],
                dest=config_key,
                required=False,
                default=argparse.SUPPRESS if config_key in {f"bot.{side}.universe.approved_coins" for side in ("long", "short")} else None,
                metavar=spec["metavar"],
                help=help_text,
            )
            if "choices" in spec:
                register_kwargs["choices"] = spec["choices"]

        visible_names = [*spec["visible"], *spec.get("command_aliases", {}).get(command, [])]
        _register_argument(
            container,
            visible_names,
            spec["hidden"],
            **register_kwargs,
        )
        visible_shorts = [
            name[1:]
            for name in visible_names
            if name.startswith("-") and not name.startswith("--")
        ]
        for short_name in visible_shorts:
            reserved_acronyms.add(short_name)
        reserved_keys.add(config_key)

    return reserved_acronyms, reserved_keys


def add_config_arguments(
    parser,
    config,
    *,
    command: Optional[str] = None,
    help_all: bool = False,
    group_map=None,
):
    """Add all CLI arguments for config parameters.

    This is the main entry point for adding config-based arguments.
    It first adds hard-coded reserved arguments (for backwards compat),
    then recursively adds remaining config parameters.

    Args:
        parser: argparse.ArgumentParser
        config: Config dict (typically from get_template_config())
    """
    reserved_acronyms, reserved_keys = add_reserved_arguments(
        parser, command=command, help_all=help_all, group_map=group_map
    )
    registered_keys = set(reserved_keys)
    add_arguments_recursively(
        parser,
        config,
        prefix="",
        acronyms=reserved_acronyms,
        skip_keys=reserved_keys,
        command=command,
        help_all=help_all,
        group_map=group_map,
        registered_keys=registered_keys,
    )
    return registered_keys


def _gpu_sizing_cli_value(value):
    """Keep automatic intent when overriding an explicitly numeric input config."""
    if value.strip().lower() == "auto":
        return "auto"
    number = float(value)
    return int(number) if number.is_integer() else number


def add_arguments_recursively(
    parser,
    config,
    prefix="",
    acronyms=None,
    skip_keys=None,
    command: Optional[str] = None,
    help_all: bool = False,
    group_map=None,
    registered_keys=None,
):
    """Recursively add CLI arguments for config parameters.

    Args:
        parser: argparse.ArgumentParser
        config: Config dict to process
        prefix: Current key prefix (e.g., "live.")
        acronyms: Set of already-used acronyms to avoid collisions
        skip_keys: Set of full config keys to skip (already added by reserved args)
    """
    if acronyms is None:
        acronyms = set()
    if skip_keys is None:
        skip_keys = set()
    if registered_keys is None:
        registered_keys = set()

    for key in sorted(config):
        value = config[key]
        full_name = f"{prefix}{key}"

        # Skip if this key was already added as a reserved argument
        if full_name in skip_keys:
            continue

        if isinstance(value, dict):
            if any(full_name.endswith(x) for x in ["approved_coins", "ignored_coins"]):
                # Handle coin dict configs as comma-separated values
                acronym = create_acronym(full_name, acronyms)
                visible_group = classify_config_argument(full_name, command, help_all)
                container = (
                    group_map.get(visible_group, parser)
                    if group_map and visible_group
                    else parser
                )
                hidden_names = [f"--{full_name.replace('.', '_')}"]
                if command is None or len(acronym) > 1:
                    hidden_names.append(f"-{acronym}")
                _register_argument(
                    container,
                    [f"--{full_name}"],
                    hidden_names,
                    type=comma_separated_values,
                    dest=full_name,
                    required=False,
                    default=None,
                    metavar="CSV",
                    help=(
                        "Override " f"{full_name}."
                        if help_all or command is None or visible_group is not None
                        else argparse.SUPPRESS
                    ),
                )
                acronyms.add(acronym)
                registered_keys.add(full_name)
                continue
            add_arguments_recursively(
                parser,
                value,
                f"{full_name}.",
                acronyms=acronyms,
                skip_keys=skip_keys,
                command=command,
                help_all=help_all,
                group_map=group_map,
                registered_keys=registered_keys,
            )
            continue
        else:
            acronym = create_acronym(full_name, acronyms)
            appendix = ""
            nullable_float_override = full_name in {
                *(f"bot.{side}.entry_cooldown.max_duration_minutes" for side in ("long", "short")),
                "backtest.ew_half_life_days",
                "bot.global.warmup.max_minutes",
            }
            nullable_coin_override = full_name in {f"bot.{side}.universe.approved_coins" for side in ("long", "short")}
            nullable_mode_override = full_name in {"live.modes.long", "live.modes.short"}
            type_ = type(value)
            if "bounds" in full_name:
                type_ = comma_separated_values_float
            if "limits" in full_name:
                type_ = str
                appendix = 'Example: "--loss_profit_ratio 0.5 --drawdown_worst 0.3333"'
            elif any(
                [
                    x in full_name
                    for x in ["approved_coins", "ignored_coins", "exchanges"]
                ]
            ):
                type_ = comma_separated_values
                appendix = "item1,item2,item3,..."
            elif "scoring" in full_name:
                type_ = comma_separated_values
                appendix = "Examples: adg,sharpe_ratio; mdg,sortino_ratio; ..."
            elif full_name in {
                "optimize.gpu.batch_size",
                "optimize.gpu.max_dispatch_candidate_bars",
            }:
                type_ = _gpu_sizing_cli_value
            elif full_name == "optimize.gpu.screening.scenarios":
                type_ = parse_screening_scenarios
                appendix = (
                    "Comma-separated labels or JSON array; [] disables screening."
                )
            elif isinstance(value, list) and "bounds" not in full_name:
                type_ = comma_separated_values
            elif full_name == "live.startup.warmup_concurrency":
                from config.control_sentinels import warmup_concurrency_cli

                type_ = warmup_concurrency_cli
            elif nullable_mode_override:
                from config.control_sentinels import forced_mode_cli

                type_ = forced_mode_cli
            elif full_name == "bot.global.warmup.max_minutes":
                from config.control_sentinels import warmup_cap_cli

                type_ = warmup_cap_cli
            elif nullable_float_override:
                type_ = optional_float
            elif value is None:
                if full_name == "backtest.btc_collateral_ltv_cap":
                    type_ = optional_float
                else:
                    type_ = str
            elif type_ == bool:
                type_ = str2bool
            elif full_name.endswith((".memory_ema_span_minutes", ".drawdown_ratio_ema_span_minutes")):
                type_ = normalize_memory_ema_span_minutes
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                type_ = float
            if "combine_ohlcvs" in full_name:
                appendix = (
                    "If true, combine ohlcvs data from all exchanges into single numpy array, otherwise backtest each exchange separately. "
                    + appendix
                )
            visible_group = classify_config_argument(full_name, command, help_all)
            container = (
                group_map.get(visible_group, parser)
                if group_map and visible_group
                else parser
            )
            hidden_names = [f"--{full_name.replace('.', '_')}"]
            if command is None or len(acronym) > 1:
                hidden_names.append(f"-{acronym}")
            _register_argument(
                container,
                [f"--{full_name}"],
                hidden_names,
                type=type_,
                dest=full_name,
                required=False,
                # Omitted nullable settings must differ from an explicit null.
                default=argparse.SUPPRESS if nullable_float_override or nullable_mode_override or nullable_coin_override else None,
                metavar=_argument_metavar(type_, full_name, value),
                help=(
                    _argument_help_text(full_name, appendix)
                    if help_all or command is None or visible_group is not None
                    else argparse.SUPPRESS
                ),
            )
            acronyms.add(acronym)
            registered_keys.add(full_name)


def recursive_config_update(config, key, value, path=None, verbose=False):
    if path is None:
        path = []

    def _coerce_value(original, new_value):
        if isinstance(original, bool):
            return bool(new_value)
        if isinstance(original, int) and not isinstance(original, bool):
            if isinstance(new_value, (int, float)):
                if isinstance(new_value, float) and not float(new_value).is_integer():
                    return float(new_value)
                return int(round(new_value))
        if isinstance(original, float):
            if isinstance(new_value, (int, float)):
                return float(new_value)
        return new_value

    key_split = key.split(".")
    current_key = key_split[0]
    full_path = ".".join(path + [current_key])

    if len(key_split) == 1:
        if current_key in config:
            coerced_value = _coerce_value(config[current_key], value)
            if coerced_value != config[current_key]:
                old_value = deepcopy(config[current_key])
                message, args = _format_config_change_message(
                    full_path, config[current_key], coerced_value
                )
                _log_config(verbose, logging.INFO, message, *args)
                config[current_key] = coerced_value
                return {
                    "path": full_path,
                    "old": old_value,
                    "new": deepcopy(coerced_value),
                }
            return None
        _log_config(verbose, logging.INFO, "added %s %s", full_path, value)
        config[current_key] = deepcopy(value)
        return {"path": full_path, "old": None, "new": deepcopy(value)}

    if current_key not in config:
        config[current_key] = {}
    elif not isinstance(config[current_key], dict):
        raise TypeError(
            f"cannot apply nested config override {'.'.join(path + key_split)!r}: "
            f"{full_path} is {type(config[current_key]).__name__}, expected dict"
        )

    return recursive_config_update(
        config[current_key],
        ".".join(key_split[1:]),
        value,
        path + [current_key],
        verbose=verbose,
    )


def effective_config_payload(config: dict) -> dict:
    """Return canonical input; normal loading validates the complete authored tree."""
    return config


def _decode_directional_cli_coin_source(value):
    """Only both-side convenience flags accept encoded directional objects."""
    import hjson

    if isinstance(value, dict):
        decoded = value
    else:
        if isinstance(value, str):
            text = value
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            text = ",".join(value)
        else:
            return value
        text = text.strip()
        if not text.startswith("{") or not text.endswith("}"):
            return value
        try:
            decoded = hjson.loads(text)
        except ValueError:
            return value  # Optional inline shorthand; canonical source validation still applies.
    if isinstance(decoded, dict):
        if set(decoded) - {"long", "short"}:
            raise ValueError("directional coin sources must contain only long/short position-side keys")
        return decoded
    return value


def update_config_with_args(
    config, args, verbose=False, allowed_keys: Optional[set[str]] = None
):
    transform_root = config
    config = effective_config_payload(config)
    from config.hsl import validate_override_paths
    nullable_overrides = {
        *(f"bot.{side}.entry_cooldown.max_duration_minutes" for side in ("long", "short")),
        "backtest.ew_half_life_days",
        "bot.global.warmup.max_minutes", "live.modes.long", "live.modes.short",
        *(f"bot.{side}.universe.approved_coins" for side in ("long", "short")),
    }
    supplied = {key: value for key, value in vars(args).items()
                if (value is not None or key in nullable_overrides)
                and (key in allowed_keys if allowed_keys is not None else "." in key or key in {"approved_coins", "ignored_coins"})}
    for field in ("approved_coins", "ignored_coins"):
        if field in supplied:
            supplied[field] = _decode_directional_cli_coin_source(supplied[field])
    if "config_version" in supplied:
        raise ValueError(
            "config_version is source schema metadata and cannot be overridden at runtime; "
            "use passivbot tool migrate-config-v9 and revalidate the configuration"
        )
    # -s/-ic deliberately set both side sources. Validate their concrete paths
    # before mutation; these convenience destinations are not config aliases.
    expanded = {}
    for key, value in supplied.items():
        if key in {"approved_coins", "ignored_coins"}:
            for side in ("long", "short"):
                expanded[f"bot.{side}.universe.{key}"] = value
        else:
            expanded[key] = value
    validate_override_paths(config, expanded, allow_engine=True)
    changed_keys = []
    diffs = []
    for key, value in vars(args).items():
        if key in {"approved_coins", "ignored_coins"}:
            value = supplied.get(key, value)
        if value is None and key not in nullable_overrides:
            continue
        if allowed_keys is not None:
            if key not in allowed_keys:
                continue
        elif "." not in key and key not in {"approved_coins", "ignored_coins"}:
            continue
        side_source_paths = {
            f"bot.{side}.universe.{field}": (side, field)
            for side in ("long", "short") for field in ("approved_coins", "ignored_coins")
        }
        if key in {"approved_coins", "ignored_coins"} or key in side_source_paths or key == "bot.global.universe.approved_coins":
            from config.universe import restore_universe_sources
            restore_universe_sources(config)
            targets = ([f"bot.{pside}.universe.{key}" for pside in ("long", "short")]
                       if key in {"approved_coins", "ignored_coins"} else [key])
            for path in targets:
                raw = value
                if key in {"approved_coins", "ignored_coins"} and isinstance(value, dict):
                    raw = value.get(path.split(".")[1], [])
                if raw == ["null"] and path.endswith("approved_coins"):
                    raw = None
                from config.universe import bind_universe_source
                raw = bind_universe_source(raw, field=path.split(".")[-1],
                    base_config_path=config.get("live", {}).get("base_config_path", ""))
                change = recursive_config_update(config, path, deepcopy(raw), verbose=verbose)
                if change:
                    changed_keys.append(path)
                    diffs.append(change)
            from config.universe import get_side_universe_sources
            config["_coins_sources"] = {field: deepcopy(get_side_universe_sources(config, field, {}))
                                       for field in ("approved_coins", "ignored_coins")}
            continue
        change = recursive_config_update(config, key, value, verbose=verbose)
        if change:
            changed_keys.append(key)
            diffs.append(change)
    if changed_keys:
        details = {"keys": changed_keys}
        if diffs:
            details["diffs"] = diffs
        record_transform(transform_root, "update_config_with_args", details)


def get_template_config():
    return get_schema_template_config()
