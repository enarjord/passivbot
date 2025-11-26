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
    aggregate_metrics,
    apply_scenario,
    build_scenarios,
    collect_suite_coin_sources,
    filter_coins_by_exchange_assignment,
    prepare_master_datasets,
    _prepare_dataset_subset,
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

    scenarios, aggregate_cfg, include_base, base_label = build_scenarios(suite_cfg)
    if include_base:
        base_scenario = SuiteScenario(
            label=base_label,
            start_date=base_start,
            end_date=base_end,
            coins=list(base_coins_list),
            ignored_coins=list(base_ignored_list),
        )
        scenarios = [base_scenario] + list(scenarios)

    suite_coin_sources = collect_suite_coin_sources(config, scenarios)

    master_coins = set(base_coins_list) if include_base else set()
    master_ignored = set(base_ignored_list) if include_base else set()
    for scenario in scenarios:
        if scenario.coins:
            master_coins.update(scenario.coins)
        if scenario.ignored_coins:
            master_ignored.update(scenario.ignored_coins)
    master_coins.update(suite_coin_sources.keys())

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
        base_config, base_exchanges, shared_array_manager=shared_array_manager
    )
    available_coins = set()
    for dataset in datasets.values():
        available_coins.update(dataset.coins)
    if not available_coins:
        raise ValueError("No coins available after preparing master datasets.")

    has_master_dataset = len(datasets) == 1 and "combined" in datasets
    if has_master_dataset:
        dataset_available_exchanges = datasets["combined"].available_exchanges
    else:
        dataset_available_exchanges = [ds.exchange for ds in datasets.values()]

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

        if has_master_dataset:
            dataset = datasets["combined"]
            allowed_exchanges = (
                list(scenario.exchanges) if scenario.exchanges else list(dataset.available_exchanges)
            )
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
            if shared_array_manager is not None:
                hlcvs_spec, _ = shared_array_manager.create_from(hlcvs_slice)
                btc_spec, _ = shared_array_manager.create_from(np.ascontiguousarray(btc_window))
            else:
                hlcvs_spec = None
                btc_spec = None

            shared_hlcvs = {} if hlcvs_spec else {dataset.exchange: hlcvs_slice}
            shared_btc = {} if btc_spec else {dataset.exchange: btc_window}
            contexts.append(
                ScenarioEvalContext(
                    label=scenario.label,
                    config=scenario_config,
                    exchanges=[dataset.exchange],
                    hlcvs_specs={dataset.exchange: hlcvs_spec},
                    btc_usd_specs={dataset.exchange: btc_spec} if btc_spec is not None else {},
                    msss={dataset.exchange: mss_slice},
                    timestamps={dataset.exchange: ts_window},
                    shared_hlcvs_np=shared_hlcvs,
                    shared_btc_np=shared_btc,
                    attachments={"hlcvs": {}, "btc": {}},
                    coin_indices={dataset.exchange: None},
                    overrides=deepcopy(scenario.overrides) if scenario.overrides else {},
                )
            )
            continue

        hlcvs_specs: Dict[str, Any] = {}
        btc_specs: Dict[str, Any] = {}
        preloaded_hlcvs: Dict[str, np.ndarray] = {}
        preloaded_btc: Dict[str, np.ndarray] = {}
        mss_slices: Dict[str, Any] = {}
        timestamps_map: Dict[str, Any] = {}
        coin_index_map: Dict[str, Optional[List[int]]] = {}
        exchanges_for_scenario: List[str] = []

        allowed_exchange_names = set(scenario.exchanges or dataset_available_exchanges)
        for exchange_key, dataset in datasets.items():
            if allowed_exchange_names and dataset.exchange not in allowed_exchange_names:
                continue
            coins_for_exchange = [coin for coin in scenario_coins if coin in dataset.coin_index]
            if not coins_for_exchange:
                continue
            exchanges_for_scenario.append(exchange_key)
            scenario_config["backtest"]["coins"][exchange_key] = list(coins_for_exchange)
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
            if shared_array_manager is not None:
                hlcvs_spec, _ = shared_array_manager.create_from(hlcvs_slice)
                hlcvs_specs[exchange_key] = hlcvs_spec
                btc_spec, _ = shared_array_manager.create_from(np.ascontiguousarray(btc_window))
                btc_specs[exchange_key] = btc_spec
            else:
                hlcvs_specs[exchange_key] = None
                preloaded_hlcvs[exchange_key] = hlcvs_slice
                btc_specs[exchange_key] = None
                preloaded_btc[exchange_key] = btc_window
            coin_index_map[exchange_key] = None

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
                hlcvs_specs=hlcvs_specs,
                btc_usd_specs=btc_specs,
                msss=mss_slices,
                timestamps=timestamps_map,
                coin_indices=coin_index_map,
                shared_hlcvs_np=dict(preloaded_hlcvs),
                shared_btc_np=dict(preloaded_btc),
                attachments={"hlcvs": {}, "btc": {}},
                overrides=deepcopy(scenario.overrides) if scenario.overrides else {},
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
        suite_override = load_config(str(suite_path), verbose=False).get("optimize", {}).get("suite")
        if suite_override is None:
            raise ValueError(f"Suite config {suite_path} must provide optimize.suite definition.")
    suite_cfg = extract_optimize_suite_config(config, suite_override)
    return suite_cfg


def summarized_metrics(
    per_scenario_metrics: Dict[str, Dict[str, float]], aggregate: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "aggregate": aggregate,
        "scenarios": per_scenario_metrics,
    }
    return payload


def extract_optimize_suite_config(
    config: Dict[str, Any], suite_override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    suite_cfg = deepcopy(config.get("optimize", {}).get("suite", {}) or {})
    if suite_override:
        suite_cfg.update(deepcopy(suite_override))
    return suite_cfg
