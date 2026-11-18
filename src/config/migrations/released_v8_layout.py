"""Structural transforms owned by the explicit v8 to v9 converter."""

import logging
from copy import deepcopy
from typing import Optional

from config.reducers import REDUCER_ALIASES, reducer_mapping_from_aliases
from config.transform_log import ConfigTransformTracker
from utils import normalize_coins_source


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    prefixed_message = "[config] " + message
    if verbose or level >= logging.WARNING:
        logging.log(level, prefixed_message, *args)
    else:
        logging.debug(prefixed_message, *args)


def migrate_suite_to_scenarios(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    backtest = result.setdefault("backtest", {})
    suite = backtest.pop("suite", None)
    if suite and isinstance(suite, dict):
        old_scenarios = suite.get("scenarios", [])
        suite_reducer, suite_reducer_present = reducer_mapping_from_aliases(
            suite,
            path="config.backtest.suite",
        )
        suite_reducer_source = next(
            (alias for alias in REDUCER_ALIASES if alias in suite),
            "aggregate",
        )
        if not suite_reducer_present:
            suite_reducer = {"default": "mean"}
        include_base = suite.get("include_base_scenario", False)
        base_label = suite.get("base_label", "base")
        suite_enabled = suite.get("enabled", False)
        if suite_enabled or old_scenarios:
            existing_reducer, existing_reducer_present = reducer_mapping_from_aliases(
                backtest,
                path="config.backtest",
            )
            if existing_reducer_present:
                for alias in REDUCER_ALIASES:
                    backtest.pop(alias, None)
            merged_reducer = {**(existing_reducer or {}), **(suite_reducer or {})}
            backtest["reducer"] = merged_reducer
            _log_config(
                verbose,
                logging.INFO,
                "migrated backtest.suite reducer -> backtest.reducer",
            )
            if tracker is not None:
                tracker.rename(
                    ["backtest", "suite", suite_reducer_source],
                    ["backtest", "reducer"],
                    merged_reducer,
                )
            new_scenarios = list(old_scenarios)
            if include_base:
                base_scenario = {"label": base_label}
                new_scenarios = [base_scenario] + new_scenarios
                _log_config(
                    verbose,
                    logging.INFO,
                    "prepended base scenario '%s' (from include_base_scenario=True)",
                    base_label,
                )
            if old_scenarios or include_base:
                backtest["scenarios"] = new_scenarios
                _log_config(
                    verbose,
                    logging.INFO,
                    "migrated backtest.suite.scenarios -> backtest.scenarios (%d scenarios)",
                    len(new_scenarios),
                )
                if tracker is not None:
                    tracker.rename(
                        ["backtest", "suite", "scenarios"],
                        ["backtest", "scenarios"],
                        new_scenarios,
                    )
        elif tracker is not None:
            tracker.remove(["backtest", "suite"], suite)

    if "combine_ohlcvs" in backtest:
        old_value = backtest.pop("combine_ohlcvs")
        _log_config(
            verbose,
            logging.INFO,
            "removed backtest.combine_ohlcvs=%s (behavior now derived from scenario exchange count)",
            old_value,
        )
        if tracker is not None:
            tracker.remove(["backtest", "combine_ohlcvs"], old_value)
    if not any(key in backtest for key in REDUCER_ALIASES):
        backtest["reducer"] = {"default": "mean"}
    backtest.setdefault("scenarios", [])
    backtest.setdefault("volume_normalization", True)


def migrate_reducer_terminology(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    backtest = result.setdefault("backtest", {})
    mapping, present = reducer_mapping_from_aliases(backtest, path="config.backtest")
    if not present:
        backtest["reducer"] = {"default": "mean"}
        if tracker is not None:
            tracker.add(["backtest", "reducer"], backtest["reducer"])
        return

    source_keys = [key for key in REDUCER_ALIASES if key in backtest]
    for key in source_keys:
        backtest.pop(key, None)
    backtest["reducer"] = mapping
    if source_keys != ["reducer"]:
        _log_config(
            verbose,
            logging.INFO,
            "normalized backtest reducer alias(es) %s -> backtest.reducer",
            ", ".join(source_keys),
        )
        if tracker is not None:
            tracker.rename(
                ["backtest", source_keys[0]],
                ["backtest", "reducer"],
                mapping,
            )


def migrate_btc_collateral_settings(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    backtest = result.setdefault("backtest", {})
    if "use_btc_collateral" in backtest:
        use_btc = backtest.pop("use_btc_collateral")
        try:
            use_btc_bool = bool(int(use_btc))
        except (TypeError, ValueError):
            use_btc_bool = bool(use_btc)
        if "btc_collateral_cap" not in backtest:
            backtest["btc_collateral_cap"] = 1.0 if use_btc_bool else 0.0
            _log_config(
                verbose,
                logging.INFO,
                "changed backtest.use_btc_collateral -> backtest.btc_collateral_cap = %s",
                backtest["btc_collateral_cap"],
            )
            if tracker is not None:
                tracker.rename(
                    ["backtest", "use_btc_collateral"],
                    ["backtest", "btc_collateral_cap"],
                    backtest["btc_collateral_cap"],
                )
        elif tracker is not None:
            tracker.remove(["backtest", "use_btc_collateral"], use_btc)
        if "btc_collateral_ltv_cap" not in backtest:
            backtest["btc_collateral_ltv_cap"] = None
            if tracker is not None:
                tracker.add(["backtest", "btc_collateral_ltv_cap"], None)

    cap = backtest.get("btc_collateral_cap")
    try:
        cap_float = float(cap)
        if tracker is not None and cap != cap_float:
            tracker.update(["backtest", "btc_collateral_cap"], cap, cap_float)
        backtest["btc_collateral_cap"] = cap_float
    except (TypeError, ValueError):
        if tracker is not None:
            tracker.update(["backtest", "btc_collateral_cap"], cap, 0.0)
        backtest["btc_collateral_cap"] = 0.0

    if "btc_collateral_ltv_cap" not in backtest:
        backtest["btc_collateral_ltv_cap"] = None
        if tracker is not None:
            tracker.add(["backtest", "btc_collateral_ltv_cap"], None)


def _coerce_legacy_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def migrate_empty_means_all_approved(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    live = result.setdefault("live", {})
    if "empty_means_all_approved" not in live:
        return

    legacy_value = live.pop("empty_means_all_approved")
    _log_config(
        verbose,
        logging.WARNING,
        "Converting released-v8 live.empty_means_all_approved; review the resulting v9 universe policies.",
    )
    if tracker is not None:
        tracker.remove(["live", "empty_means_all_approved"], legacy_value)

    if not _coerce_legacy_bool(legacy_value):
        return

    approved_source = deepcopy(live.get("approved_coins"))
    explicit_side_source = isinstance(approved_source, dict) and any(
        key in approved_source for key in ("long", "short")
    )
    normalized = normalize_coins_source(approved_source, allow_all=True)
    if explicit_side_source or any(normalized[pside] for pside in ("long", "short")):
        return

    live["approved_coins"] = "all"
    _log_config(
        verbose,
        logging.WARNING,
        "migrated legacy live.empty_means_all_approved=true with empty live.approved_coins "
        "to live.approved_coins='all'",
    )
    if tracker is not None:
        tracker.update(["live", "approved_coins"], approved_source, "all")


def apply_released_v8_layout_transforms(result: dict) -> None:
    from .churn_gate import migrate_initial_entry_distance_gate
    from .gpu_population import migrate_gpu_population
    from .gpu_screening import migrate_gpu_screening
    from .renames import apply_backward_compatibility_renames, rename_config_keys

    migrate_gpu_screening(result)
    migrate_gpu_population(result)
    migrate_initial_entry_distance_gate(result, verbose=False)
    apply_backward_compatibility_renames(result, verbose=False)
    migrate_btc_collateral_settings(result, verbose=False)
    migrate_empty_means_all_approved(result, verbose=False)
    migrate_suite_to_scenarios(result, verbose=False)
    migrate_reducer_terminology(result, verbose=False)
    rename_config_keys(result, verbose=False)
