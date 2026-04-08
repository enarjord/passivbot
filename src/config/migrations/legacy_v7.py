import logging
from copy import deepcopy
from typing import Optional

from config.schema import CONFIG_SCHEMA_VERSION
from config.transform_log import ConfigTransformTracker
from utils import normalize_coins_source


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    prefixed_message = "[config] " + message
    if verbose or level >= logging.WARNING:
        logging.log(level, prefixed_message, *args)
    else:
        logging.debug(prefixed_message, *args)


def _parse_version_tuple(value: object) -> Optional[tuple[int, ...]]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    parts = normalized.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def migrate_config_version(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    current_version = result.get("config_version")
    current_parsed = _parse_version_tuple(current_version)
    target_parsed = _parse_version_tuple(CONFIG_SCHEMA_VERSION)
    if target_parsed is None:
        raise ValueError(f"internal error: invalid CONFIG_SCHEMA_VERSION {CONFIG_SCHEMA_VERSION!r}")
    if current_version == CONFIG_SCHEMA_VERSION:
        return
    if current_version in (None, ""):
        _log_config(
            verbose,
            logging.INFO,
            "pre-versioned config detected. attempting migration to schema %s",
            CONFIG_SCHEMA_VERSION,
        )
    elif current_parsed is None:
        raise ValueError(
            f"config.config_version must be a semantic version like {CONFIG_SCHEMA_VERSION}; "
            f"got {current_version!r}"
        )
    elif current_parsed > target_parsed:
        raise ValueError(
            f"config.config_version {current_version!r} is newer than supported schema "
            f"{CONFIG_SCHEMA_VERSION}; upgrade Passivbot"
        )
    else:
        _log_config(
            verbose,
            logging.INFO,
            "%s config detected. attempting migration to schema %s",
            current_version,
            CONFIG_SCHEMA_VERSION,
        )
    result["config_version"] = CONFIG_SCHEMA_VERSION
    if tracker is not None:
        if current_version in (None, ""):
            tracker.add(["config_version"], CONFIG_SCHEMA_VERSION)
        else:
            tracker.update(["config_version"], current_version, CONFIG_SCHEMA_VERSION)


def migrate_suite_to_scenarios(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    backtest = result.setdefault("backtest", {})
    suite = backtest.pop("suite", None)
    if suite and isinstance(suite, dict):
        old_scenarios = suite.get("scenarios", [])
        aggregate = suite.get("aggregate", {"default": "mean"})
        include_base = suite.get("include_base_scenario", False)
        base_label = suite.get("base_label", "base")
        suite_enabled = suite.get("enabled", False)
        if suite_enabled or old_scenarios:
            existing_aggregate = backtest.get("aggregate", {})
            merged_aggregate = {**existing_aggregate, **aggregate}
            backtest["aggregate"] = merged_aggregate
            _log_config(verbose, logging.INFO, "migrated backtest.suite.aggregate -> backtest.aggregate")
            if tracker is not None:
                tracker.rename(["backtest", "suite", "aggregate"], ["backtest", "aggregate"], merged_aggregate)
            new_scenarios = list(old_scenarios)
            if include_base:
                base_scenario = {"label": base_label}
                new_scenarios = [base_scenario] + new_scenarios
                _log_config(verbose, logging.INFO, "prepended base scenario '%s' (from include_base_scenario=True)", base_label)
            if old_scenarios or include_base:
                backtest["scenarios"] = new_scenarios
                _log_config(verbose, logging.INFO, "migrated backtest.suite.scenarios -> backtest.scenarios (%d scenarios)", len(new_scenarios))
                if tracker is not None:
                    tracker.rename(["backtest", "suite", "scenarios"], ["backtest", "scenarios"], new_scenarios)
        elif tracker is not None:
            tracker.remove(["backtest", "suite"], suite)

    if "combine_ohlcvs" in backtest:
        old_value = backtest.pop("combine_ohlcvs")
        _log_config(verbose, logging.INFO, "removed backtest.combine_ohlcvs=%s (behavior now derived from scenario exchange count)", old_value)
        if tracker is not None:
            tracker.remove(["backtest", "combine_ohlcvs"], old_value)
    backtest.setdefault("aggregate", {"default": "mean"})
    backtest.setdefault("scenarios", [])
    backtest.setdefault("volume_normalization", True)


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
        "live.empty_means_all_approved is deprecated and will be removed in a future release; "
        "legacy configs still work for now. Prefer explicit live.approved_coins='all' or per-side "
        "'all' entries in new configs.",
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
