import argparse
import logging
import re
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
    HSL_COOLDOWN_POSITION_POLICIES,
    HSL_SIGNAL_MODES,
    MONITOR_BOOL_KEYS,
    PYMOO_ALGORITHMS,
    PYMOO_REF_DIR_METHODS,
    normalize_hsl_cooldown_position_policy,
    normalize_hsl_signal_mode,
)
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
from config.normalize import normalize_config
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
from config.schema import get_template_config as get_schema_template_config
from config.tree_ops import (
    add_missing_keys_recursively,
    remove_unused_keys_recursively,
)
from config.transform_log import ConfigTransformTracker, record_transform
from config.logging_summary import emit_transform_summary
from config.migrations import (
    apply_backward_compatibility_renames as apply_migration_renames,
    build_base_config_from_flavor as build_migration_base_config_from_flavor,
    detect_flavor as detect_migration_flavor,
    migrate_btc_collateral_settings as migrate_btc_collateral_settings_v7,
    migrate_suite_to_scenarios as migrate_suite_to_scenarios_v7,
    rename_config_keys as rename_migration_config_keys,
)
from pure_funcs import sort_dict_keys, str2bool
from utils import dump_json_streamlined, normalize_coins_source, symbol_to_coin


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    log_config_message(verbose, level, message, *args)


Path = Tuple[str, ...]  # ("bot", "long", "entry_grid_spacing_pct")
HSL_TIER_RATIO_KEYS = ("yellow", "orange")
HSL_PSIDE_KEYS = (
    "hsl_enabled",
    "hsl_red_threshold",
    "hsl_ema_span_minutes",
    "hsl_cooldown_minutes_after_red",
    "hsl_no_restart_drawdown_threshold",
    "hsl_orange_tier_mode",
    "hsl_panic_close_order_type",
    "hsl_tier_ratios",
)


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
    config_copy = deepcopy(config)
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


def apply_allowed_modifications(src, modifications, allowed_overrides, return_full=True):
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
    return staged_load_override_config(config, coin, config_loader=lambda path: load_config(path, verbose=False))


def parse_old_coin_flags(config) -> dict:
    return staged_parse_old_coin_flags(config)


def _apply_backward_compatibility_renames(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    apply_migration_renames(result, verbose=verbose, tracker=tracker)


def _migrate_suite_to_scenarios(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    migrate_suite_to_scenarios_v7(result, verbose=verbose, tracker=tracker)


def _migrate_btc_collateral_settings(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    migrate_btc_collateral_settings_v7(result, verbose=verbose, tracker=tracker)


def detect_flavor(config: dict, template: dict) -> str:
    return detect_migration_flavor(config, template)


def build_base_config_from_flavor(config: dict, template: dict, flavor: str, verbose: bool) -> dict:
    return build_migration_base_config_from_flavor(config, template, flavor, verbose)


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
    tracker: Optional[ConfigTransformTracker] = None,
) -> dict:
    return staged_format_bot_config(
        bot_cfg,
        live_cfg=live_cfg,
        verbose=verbose,
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
    staged_hydrate_missing_template_fields(template, result, verbose=verbose, tracker=tracker)


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


def format_config(config: dict, verbose=True, live_only=False, base_config_path: str = "") -> dict:
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
            cleaned = {}
            for key, value in source_dict.items():
                if key in template_node:
                    cleaned[key] = _clean_with_template(template_node[key], value, path + (key,))
                else:
                    cleaned[key] = _clean_dynamic_node(value)
            for key, tmpl_value in template_node.items():
                if key not in cleaned:
                    cleaned[key] = _clean_with_template(tmpl_value, None, path + (key,))
            return cleaned
        if not template_node:
            return _clean_dynamic_node(source_dict)
        result = {}
        for key, tmpl_value in template_node.items():
            result[key] = _clean_with_template(tmpl_value, source_dict.get(key), path + (key,))
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
    template = get_template_config()
    cleaned = _clean_with_template(template, config or {})
    return sort_dict_keys(cleaned)


def strip_config_metadata(config: dict, *, keys: Iterable[str] | None = None) -> dict:
    """
    Return a deep-copied config with the provided metadata keys removed recursively.
    Defaults to removing `_raw`, `_raw_effective`, and `_transform_log`.
    """

    removal = set(keys or ("_raw", "_raw_effective", "_transform_log", "_coins_sources"))

    def _strip(node):
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items() if k not in removal}
        if isinstance(node, list):
            return [_strip(item) for item in node]
        return deepcopy(node)

    return _strip(config)


def _limits_structurally_equal(raw_limits: Any, normalized_limits: List[Dict[str, Any]]) -> bool:
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
#       "commands": {"live", "backtest", "optimize"},
#       "group": {"live": "Coin Selection", ...},
#       "type": type_converter,
#       "metavar": "CSV|INT|...",
#       "help": "Human-facing help text",
#   }
RESERVED_CLI_ARGS = {
    "live.approved_coins": {
        "visible": ["--symbols", "-s"],
        "hidden": ["--live.approved_coins", "--live_approved_coins"],
        "type": comma_separated_values,
        "metavar": "CSV_OR_PATH",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": (
            "Approved coins. Comma-separated coins like BTC,ETH,XRP, or path to a JSON "
            "coin list file. Use coin tickers, not exchange symbols."
        ),
    },
    "live.ignored_coins": {
        "visible": ["--ignored-coins", "-ic"],
        "hidden": ["--live.ignored_coins", "--live_ignored_coins"],
        "type": comma_separated_values,
        "metavar": "CSV_OR_PATH",
        "commands": {"live", "backtest"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
        },
        "help": "Ignored coins. Comma-separated coins or path to a JSON coin list file.",
    },
    "live.minimum_coin_age_days": {
        "visible": ["--minimum-coin-age-days", "-mcad"],
        "hidden": ["--live.minimum_coin_age_days", "--live_minimum_coin_age_days"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "optimize"},
        "group": {
            "live": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Minimum coin age in days required before a coin is eligible to trade.",
    },
    "live.filter_by_min_effective_cost": {
        "visible": ["--filter-by-min-effective-cost", "-fbmec"],
        "hidden": [
            "--live.filter_by_min_effective_cost",
            "--live_filter_by_min_effective_cost",
        ],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Filter out coins whose minimum effective cost exceeds the configured size.",
    },
    "live.hedge_mode": {
        "visible": ["--hedge-mode", "-hm"],
        "hidden": ["--live.hedge_mode", "--live_hedge_mode"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live", "optimize"},
        "group": {"live": "Behavior"},
        "help": (
            "Enable or disable hedge mode. If the exchange does not support simultaneous "
            "long and short on the same coin, the bot will use hedge_mode=false."
        ),
    },
    "live.leverage": {
        "visible": ["--leverage", "-lev"],
        "hidden": ["--live.leverage", "--live_leverage"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Set leverage for this run.",
    },
    "live.market_orders_allowed": {
        "visible": ["--market-orders-allowed", "-moa"],
        "hidden": ["--live.market_orders_allowed", "--live_market_orders_allowed"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Allow or disallow market orders.",
    },
    "live.max_realized_loss_pct": {
        "visible": ["--max-realized-loss-pct", "-mrlp"],
        "hidden": ["--live.max_realized_loss_pct", "--live_max_realized_loss_pct"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live", "optimize"},
        "group": {"live": "Behavior"},
        "help": "Maximum realized loss percentage allowed before trading is halted.",
    },
    "live.price_distance_threshold": {
        "visible": ["--price-distance-threshold", "-pdt"],
        "hidden": ["--live.price_distance_threshold", "--live_price_distance_threshold"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Reject orders whose price is too far from the market.",
    },
    "live.time_in_force": {
        "visible": ["--time-in-force", "-tif"],
        "hidden": ["--live.time_in_force", "--live_time_in_force"],
        "type": str,
        "metavar": "VALUE",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Time-in-force policy for live orders.",
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
        "type": float,
        "metavar": "FLOAT",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Backtest Runtime",
            "optimize": "Date Range",
        },
        "help": "Backtest candle interval in minutes.",
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
    "backtest.aggregate.default": {
        "visible": ["--aggregate-default"],
        "hidden": ["--backtest.aggregate.default", "--backtest_aggregate_default"],
        "type": str,
        "metavar": "VALUE",
        "commands": {"backtest"},
        "group": {"backtest": "Suite"},
        "help": "Suite-only: default aggregation to use for scenario metrics.",
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
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer backend to use. Supported values: deap or pymoo.",
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
}

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
        "Optimize DEAP",
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


def _argument_help_text(full_name: str, appendix: str) -> str:
    base = f"Override {full_name}."
    if appendix:
        return f"{base} {appendix}".strip()
    return base


def _classify_live_argument(full_name: str, help_all: bool) -> Optional[str]:
    coin_selection = {
        "live.approved_coins",
        "live.ignored_coins",
        "live.minimum_coin_age_days",
    }
    behavior = {
        "live.filter_by_min_effective_cost",
        "live.forced_mode_long",
        "live.forced_mode_short",
        "live.hedge_mode",
        "live.leverage",
        "live.market_orders_allowed",
        "live.max_realized_loss_pct",
        "live.order_match_tolerance_pct",
        "live.price_distance_threshold",
    }
    runtime = {
        "live.execution_delay_seconds",
        "live.max_concurrent_api_requests",
        "live.max_n_restarts_per_day",
        "live.max_ohlcv_fetches_per_minute",
        "live.recv_window_ms",
        "live.time_in_force",
        "live.user",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in behavior:
        return "Behavior"
    if full_name in runtime:
        return "Runtime"
    return "Advanced Overrides" if help_all else None


def _classify_backtest_argument(full_name: str, help_all: bool) -> Optional[str]:
    coin_selection = {
        "backtest.exchanges",
        "live.approved_coins",
        "live.ignored_coins",
        "live.minimum_coin_age_days",
    }
    date_range = {
        "backtest.end_date",
        "backtest.max_warmup_minutes",
        "backtest.start_date",
    }
    runtime = {
        "backtest.aggregate.default",
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
    coin_selection = {
        "backtest.exchanges",
        "live.approved_coins",
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
        "optimize.pareto_max_size",
        "optimize.population_size",
        "optimize.scoring",
    }
    backtest_runtime = {
        "backtest.aggregate.default",
        "backtest.balance_sample_divider",
        "backtest.btc_collateral_cap",
        "backtest.btc_collateral_ltv_cap",
        "backtest.compress_cache",
        "backtest.dynamic_wel_by_tradability",
        "backtest.filter_by_min_effective_cost",
        "backtest.gap_tolerance_ohlcvs_minutes",
        "backtest.maker_fee_override",
        "backtest.max_warmup_minutes",
        "backtest.ohlcv_source_dir",
        "backtest.starting_balance",
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
    optimize_deap = {
        "optimize.crossover_eta",
        "optimize.crossover_probability",
        "optimize.mutation_eta",
        "optimize.mutation_indpb",
        "optimize.mutation_probability",
        "optimize.offspring_multiplier",
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
    if full_name in optimize_deap or full_name.startswith("optimize.deap.shared."):
        return "Optimize DEAP" if help_all else None
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
            spec.get("group", {}).get(command)
            if command is not None
            else None
        )
        container = group_map.get(visible_group, parser) if group_map and visible_group else parser

        register_kwargs = dict(
            type=spec["type"],
            dest=config_key,
            required=False,
            default=None,
            metavar=spec["metavar"],
            help=(
                spec["help"]
                if help_all or visible_group is not None or command is None
                else argparse.SUPPRESS
            ),
        )
        if "choices" in spec:
            register_kwargs["choices"] = spec["choices"]

        _register_argument(
            container,
            spec["visible"],
            spec["hidden"],
            **register_kwargs,
        )
        visible_shorts = [name[1:] for name in spec["visible"] if name.startswith("-") and not name.startswith("--")]
        for short_name in visible_shorts:
            reserved_acronyms.add(short_name)
        reserved_keys.add(config_key)

    return reserved_acronyms, reserved_keys


def add_config_arguments(parser, config, *, command: Optional[str] = None, help_all: bool = False, group_map=None):
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
    add_arguments_recursively(
        parser,
        config,
        prefix="",
        acronyms=reserved_acronyms,
        skip_keys=reserved_keys,
        command=command,
        help_all=help_all,
        group_map=group_map,
    )


def add_arguments_recursively(
    parser,
    config,
    prefix="",
    acronyms=None,
    skip_keys=None,
    command: Optional[str] = None,
    help_all: bool = False,
    group_map=None,
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
                    group_map.get(visible_group, parser) if group_map and visible_group else parser
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
                        "Override "
                        f"{full_name}."
                        if help_all or command is None
                        else argparse.SUPPRESS
                    ),
                )
                acronyms.add(acronym)
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
            )
            continue
        else:
            acronym = create_acronym(full_name, acronyms)
            appendix = ""
            type_ = type(value)
            if "bounds" in full_name:
                type_ = comma_separated_values_float
            if "limits" in full_name:
                type_ = str
                appendix = 'Example: "--loss_profit_ratio 0.5 --drawdown_worst 0.3333"'
            elif any([x in full_name for x in ["approved_coins", "ignored_coins", "exchanges"]]):
                type_ = comma_separated_values
                appendix = "item1,item2,item3,..."
            elif "scoring" in full_name:
                type_ = comma_separated_values
                appendix = "Examples: adg,sharpe_ratio; mdg,sortino_ratio; ..."
            elif value is None:
                if full_name == "backtest.btc_collateral_ltv_cap":
                    type_ = optional_float
                else:
                    type_ = str
            elif type_ == bool:
                type_ = str2bool
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                type_ = float
            if "combine_ohlcvs" in full_name:
                appendix = (
                    "If true, combine ohlcvs data from all exchanges into single numpy array, otherwise backtest each exchange separately. "
                    + appendix
                )
            visible_group = classify_config_argument(full_name, command, help_all)
            container = group_map.get(visible_group, parser) if group_map and visible_group else parser
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
                default=None,
                metavar=_argument_metavar(type_, full_name, value),
                help=(
                    _argument_help_text(full_name, appendix)
                    if help_all or command is None
                    else argparse.SUPPRESS
                ),
            )
            acronyms.add(acronym)


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

    if key in config:
        coerced_value = _coerce_value(config[key], value)
        if coerced_value != config[key]:
            full_path = ".".join(path + [key])
            old_value = deepcopy(config[key])
            _log_config(
                verbose, logging.INFO, "changed %s %s -> %s", full_path, config[key], coerced_value
            )
            config[key] = coerced_value
            return {"path": full_path, "old": old_value, "new": deepcopy(coerced_value)}
        return None

    key_split = key.split(".")
    if key_split[0] in config:
        new_path = path + [key_split[0]]
        return recursive_config_update(
            config[key_split[0]], ".".join(key_split[1:]), value, new_path, verbose=verbose
        )

    return None


def update_config_with_args(config, args, verbose=False):
    changed_keys = []
    diffs = []
    for key, value in vars(args).items():
        if value is None:
            continue
        if key in {"live.approved_coins", "live.ignored_coins"}:
            normalized = normalize_coins_source(value)
            change = recursive_config_update(config, key, normalized, verbose=verbose)
            source_key = key.split(".")[-1]
            config.setdefault("_coins_sources", {})[source_key] = deepcopy(normalized)
            if change:
                changed_keys.append(key)
                diffs.append(change)
            continue
        change = recursive_config_update(config, key, value, verbose=verbose)
        if change:
            changed_keys.append(key)
            diffs.append(change)
    if changed_keys:
        details = {"keys": changed_keys}
        if diffs:
            details["diffs"] = diffs
        record_transform(config, "update_config_with_args", details)


def get_template_config():
    return get_schema_template_config()
