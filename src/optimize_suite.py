"""Optimizer helpers for running suites of backtests per candidate."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config_utils import (
    load_config,
    parse_overrides,
    require_config_value,
    require_live_value,
    format_config,
)
from shared_arrays import attach_shared_array
from suite_runner import (
    SuiteScenario,
    extract_suite_config,
    aggregate_metrics,
    apply_scenario,
    build_scenarios,
    collect_suite_coin_sources,
    filter_coins_by_exchange_assignment,
    prepare_master_datasets,
    _prepare_dataset_subset,
    _compute_slice_indices,
    _determine_needed_individual_exchanges,
)
from utils import format_approved_ignored_coins, load_markets, ts_to_date, utc_ms


@dataclass
class ScenarioEvalContext:
    label: str
    config: Dict[str, Any]
    exchanges: List[str]
    hlcvs_specs: Dict[str, Any]
    btc_usd_specs: Dict[str, Any]
    msss: Dict[str, Any]
    timestamps: Dict[str, Any]
    shared_hlcvs_np: Dict[str, np.ndarray]
    shared_btc_np: Dict[str, np.ndarray]
    attachments: Dict[str, Dict[str, Any]]
    coin_indices: Dict[str, Optional[List[int]]]
    overrides: Dict[str, Any]
    # Slice metadata for lazy slicing from master dataset (memory optimization)
    master_hlcvs_specs: Optional[Dict[str, Any]] = None
    master_btc_specs: Optional[Dict[str, Any]] = None
    time_slice: Optional[tuple] = None  # (start_idx, end_idx)
    coin_slice_indices: Optional[Dict[str, List[int]]] = None  # per-exchange coin indices


async def prepare_suite_contexts(
    config: Dict[str, Any],
    suite_cfg: Dict[str, Any],
    *,
    shared_array_manager,
) -> tuple[List[ScenarioEvalContext], Dict[str, Any]]:
    """Prepare datasets and configs for every optimizer suite scenario."""

    base_exchanges = require_config_value(config, "backtest.exchanges")
    for exchange in base_exchanges:
        await load_markets(exchange, verbose=False)
    await format_approved_ignored_coins(config, base_exchanges, verbose=False)

    base_start = require_config_value(config, "backtest.start_date")
    base_end = require_config_value(config, "backtest.end_date")
    base_coins = require_live_value(config, "approved_coins")
    if isinstance(base_coins, dict):
        coins_union = set(base_coins.get("long", [])) | set(base_coins.get("short", []))
        base_coins_list = sorted(coins_union)
    elif isinstance(base_coins, list):
        base_coins_list = list(base_coins)
    else:
        base_coins_list = []

    base_ignored = require_live_value(config, "ignored_coins")
    if isinstance(base_ignored, dict):
        ignored_union = set(base_ignored.get("long", [])) | set(base_ignored.get("short", []))
        base_ignored_list = sorted(ignored_union)
    elif isinstance(base_ignored, list):
        base_ignored_list = list(base_ignored)
    else:
        base_ignored_list = []

    scenarios, aggregate_cfg = build_scenarios(suite_cfg, base_exchanges=base_exchanges)

    # Determine which individual exchange datasets are needed for single-exchange scenarios
    needed_individual = _determine_needed_individual_exchanges(scenarios, base_exchanges)

    suite_coin_sources = collect_suite_coin_sources(config, scenarios)

    # Collect all coins from scenarios (or use base if no scenario-specific coins)
    master_coins = set()
    master_ignored = set()
    for scenario in scenarios:
        if scenario.coins:
            master_coins.update(scenario.coins)
        if scenario.ignored_coins:
            master_ignored.update(scenario.ignored_coins)
    master_coins.update(suite_coin_sources.keys())

    # If no scenarios define explicit coins, fall back to base_coins_list
    if not master_coins and base_coins_list:
        logging.info(
            "No scenario-specific coins found; using base approved_coins: %s",
            base_coins_list,
        )
        master_coins = set(base_coins_list)
    if not master_ignored and base_ignored_list:
        master_ignored = set(base_ignored_list)

    master_coins_list = sorted(master_coins)
    master_ignored_list = sorted(master_ignored)

    base_config = deepcopy(config)
    if isinstance(base_config["live"]["approved_coins"], dict):
        base_config["live"]["approved_coins"]["long"] = master_coins_list
        base_config["live"]["approved_coins"]["short"] = master_coins_list
    else:
        base_config["live"]["approved_coins"] = master_coins_list
    if isinstance(base_config["live"]["ignored_coins"], dict):
        base_config["live"]["ignored_coins"]["long"] = master_ignored_list
        base_config["live"]["ignored_coins"]["short"] = master_ignored_list
    else:
        base_config["live"]["ignored_coins"] = master_ignored_list
    base_config["backtest"]["coins"] = {}
    base_config["backtest"]["coin_sources"] = suite_coin_sources

    datasets = await prepare_master_datasets(
        base_config,
        base_exchanges,
        shared_array_manager=shared_array_manager,
        needed_individual_exchanges=needed_individual,
    )
    available_coins = set()
    for dataset in datasets.values():
        available_coins.update(dataset.coins)
    if not available_coins:
        raise ValueError("No coins available after preparing master datasets.")

    has_combined = "combined" in datasets
    # Available exchanges exclude "combined" pseudo-exchange
    dataset_available_exchanges = sorted(
        set(ds.exchange for ds in datasets.values() if ds.exchange != "combined")
    ) or (datasets["combined"].available_exchanges if has_combined else [])

    contexts: List[ScenarioEvalContext] = []

    for scenario in scenarios:
        try:
            scenario_config_raw, scenario_coins = apply_scenario(
                base_config,
                scenario,
                master_coins=master_coins_list,
                master_ignored=master_ignored_list,
                available_exchanges=dataset_available_exchanges,
                available_coins=available_coins,
                base_coin_sources=suite_coin_sources,
            )
        except ValueError as exc:
            logging.warning("Skipping scenario %s: %s", scenario.label, exc)
            continue
        scenario_config = format_config(scenario_config_raw, verbose=False)
        scenario_config = parse_overrides(scenario_config, verbose=False)
        scenario_config.setdefault("backtest", {})
        scenario_config["backtest"]["coins"] = {}
        # Debug visibility to ensure scenario-specific windows/overrides are honored
        logging.debug(
            "Suite scenario %s | start=%s end=%s coins=%s overrides=%s",
            scenario.label,
            scenario_config["backtest"].get("start_date"),
            scenario_config["backtest"].get("end_date"),
            list(scenario_config["backtest"].get("coins", {}).keys())
            or scenario_config["backtest"].get("coins"),
            bool(scenario.overrides),
        )

        # Determine which dataset(s) to use based on scenario's exchange restriction
        raw_scenario_exchanges = set(scenario.exchanges) if scenario.exchanges else None
        all_exchanges_set = set(dataset_available_exchanges)

        # Filter scenario exchanges to only those actually available in the dataset
        if raw_scenario_exchanges:
            unavailable = raw_scenario_exchanges - all_exchanges_set
            if unavailable:
                logging.debug(
                    "Scenario %s: exchanges %s not available in dataset, using %s",
                    scenario.label,
                    sorted(unavailable),
                    sorted(raw_scenario_exchanges & all_exchanges_set) or "all available",
                )
            scenario_exchanges = raw_scenario_exchanges & all_exchanges_set
            if not scenario_exchanges:
                # If no overlap, fall back to all available exchanges
                scenario_exchanges = all_exchanges_set
        else:
            scenario_exchanges = all_exchanges_set

        # Use combined dataset when:
        # 1. It exists, AND
        # 2. Scenario uses all available exchanges (or doesn't restrict)
        use_combined = has_combined and scenario_exchanges == all_exchanges_set

        if use_combined:
            dataset = datasets["combined"]
            allowed_exchanges = list(dataset.available_exchanges)
            selected_coins, skipped_coins = filter_coins_by_exchange_assignment(
                scenario_coins,
                allowed_exchanges,
                dataset.coin_exchange,
                default_exchange=dataset.exchange,
            )
            if skipped_coins:
                logging.warning(
                    "Scenario %s: skipping %d coin(s) outside allowed exchanges (%s): %s",
                    scenario.label,
                    len(skipped_coins),
                    ",".join(allowed_exchanges),
                    ",".join(skipped_coins[:10]),
                )
            if not selected_coins:
                logging.warning(
                    "Skipping scenario %s: no coins remain after exchange filtering.",
                    scenario.label,
                )
                continue
            # Create per-scenario SharedMemory (sliced from master)
            # This is shared by ALL workers - memory scales with scenarios, not workers
            (
                hlcvs_slice,
                btc_window,
                ts_window,
                mss_slice,
            ) = _prepare_dataset_subset(
                dataset,
                scenario_config,
                selected_coins,
                scenario.label,
            )
            scenario_config["backtest"]["coins"][dataset.exchange] = list(selected_coins)

            # Create SharedMemory for this scenario's sliced data
            hlcvs_spec, _ = shared_array_manager.create_from(
                np.ascontiguousarray(hlcvs_slice, dtype=np.float64)
            )
            btc_spec = None
            if btc_window is not None:
                btc_spec, _ = shared_array_manager.create_from(
                    np.ascontiguousarray(btc_window, dtype=np.float64)
                )
            # Free the slice arrays after copying to SharedMemory
            del hlcvs_slice, btc_window

            contexts.append(
                ScenarioEvalContext(
                    label=scenario.label,
                    config=scenario_config,
                    exchanges=[dataset.exchange],
                    hlcvs_specs={dataset.exchange: hlcvs_spec},
                    btc_usd_specs={dataset.exchange: btc_spec},
                    msss={dataset.exchange: mss_slice},
                    timestamps={dataset.exchange: ts_window},
                    shared_hlcvs_np={},
                    shared_btc_np={},
                    attachments={"hlcvs": {}, "btc": {}},
                    coin_indices={dataset.exchange: None},  # Already sliced
                    overrides=deepcopy(scenario.overrides) if scenario.overrides else {},
                    master_hlcvs_specs=None,
                    master_btc_specs=None,
                    time_slice=None,
                    coin_slice_indices=None,
                )
            )
            continue

        mss_slices: Dict[str, Any] = {}
        timestamps_map: Dict[str, Any] = {}
        hlcvs_specs_map: Dict[str, Any] = {}
        btc_specs_map: Dict[str, Any] = {}
        exchanges_for_scenario: List[str] = []

        # Use per-exchange datasets for scenarios with exchange restrictions
        allowed_exchange_names = set(scenario.exchanges or dataset_available_exchanges)
        for exchange_key, dataset in datasets.items():
            # Skip "combined" pseudo-dataset; use actual exchange datasets
            if exchange_key == "combined":
                continue
            if allowed_exchange_names and dataset.exchange not in allowed_exchange_names:
                continue
            coins_for_exchange = [coin for coin in scenario_coins if coin in dataset.coin_index]
            if not coins_for_exchange:
                continue
            exchanges_for_scenario.append(exchange_key)
            scenario_config["backtest"]["coins"][exchange_key] = list(coins_for_exchange)

            # Create per-scenario SharedMemory (sliced from master)
            (
                hlcvs_slice,
                btc_window,
                ts_window,
                mss_slice,
            ) = _prepare_dataset_subset(
                dataset,
                scenario_config,
                coins_for_exchange,
                scenario.label,
            )
            # Create SharedMemory for this scenario's sliced data
            hlcvs_spec, _ = shared_array_manager.create_from(
                np.ascontiguousarray(hlcvs_slice, dtype=np.float64)
            )
            btc_spec = None
            if btc_window is not None:
                btc_spec, _ = shared_array_manager.create_from(
                    np.ascontiguousarray(btc_window, dtype=np.float64)
                )
            # Free slice arrays after copying to SharedMemory
            del hlcvs_slice, btc_window

            hlcvs_specs_map[exchange_key] = hlcvs_spec
            btc_specs_map[exchange_key] = btc_spec
            mss_slices[exchange_key] = mss_slice
            timestamps_map[exchange_key] = ts_window

        if not exchanges_for_scenario:
            logging.warning("Skipping scenario %s: no exchanges after filtering.", scenario.label)
            continue

        contexts.append(
            ScenarioEvalContext(
                label=scenario.label,
                config=scenario_config,
                exchanges=exchanges_for_scenario,
                hlcvs_specs=hlcvs_specs_map,
                btc_usd_specs=btc_specs_map,
                msss=mss_slices,
                timestamps=timestamps_map,
                coin_indices={ex: None for ex in exchanges_for_scenario},  # Already sliced
                shared_hlcvs_np={},
                shared_btc_np={},
                attachments={"hlcvs": {}, "btc": {}},
                overrides=deepcopy(scenario.overrides) if scenario.overrides else {},
                master_hlcvs_specs=None,
                master_btc_specs=None,
                time_slice=None,
                coin_slice_indices=None,
            )
        )

    if not contexts:
        raise ValueError("Suite configuration produced no runnable scenarios after filtering.")

    return contexts, aggregate_cfg


def ensure_suite_config(config_path: Path, suite_path: Optional[Path]) -> Dict[str, Any]:
    config = load_config(str(config_path), verbose=False)
    config = parse_overrides(config, verbose=False)
    suite_override = None
    if suite_path:
        override_config = load_config(str(suite_path), verbose=False)
        override_backtest = override_config.get("backtest", {})
        # Support both new (scenarios at top level) and legacy (suite wrapper) formats
        if "scenarios" in override_backtest:
            suite_override = {
                "scenarios": override_backtest.get("scenarios", []),
                "aggregate": override_backtest.get("aggregate", {"default": "mean"}),
            }
        elif "suite" in override_backtest:
            # Legacy format - extract from suite wrapper
            suite_override = override_backtest["suite"]
        else:
            raise ValueError(f"Suite config {suite_path} must provide backtest.scenarios definition.")
    return extract_suite_config(config, suite_override)


def summarized_metrics(
    per_scenario_metrics: Dict[str, Dict[str, float]], aggregate: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "aggregate": aggregate,
        "scenarios": per_scenario_metrics,
    }
    return payload


#
# Suite configuration is now canonical under backtest.scenarios.
# Optimizer suite uses the same schema and reads it via suite_runner.extract_suite_config().
#
