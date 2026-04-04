import logging
from typing import Optional

from config.transform_log import ConfigTransformTracker


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
