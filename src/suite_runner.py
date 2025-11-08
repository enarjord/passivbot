"""
Shared helpers for running backtest/optimizer suites.

The suite runner prepares shared datasets, applies scenario overrides, and
invokes the existing backtest pipeline for every scenario.  Both the CLI
backtester and the optimizer import this module when operating in suite mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config_utils import (
    format_config,
    load_config,
    parse_overrides,
    require_config_value,
    require_live_value,
)
from logging_setup import configure_logging
from main import manage_rust_compilation
from utils import format_approved_ignored_coins, load_markets, ts_to_date, utc_ms


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SuiteScenario:
    label: str
    start_date: Optional[str]
    end_date: Optional[str]
    coins: Optional[List[str]]
    ignored_coins: Optional[List[str]]
    exchanges: Optional[List[str]] = None
    coin_sources: Optional[Dict[str, str]] = None


@dataclass
class ScenarioResult:
    scenario: SuiteScenario
    per_exchange: Dict[str, Dict[str, Any]]
    combined_metrics: Dict[str, float]
    elapsed_seconds: float
    output_path: Optional[Path]


@dataclass
class SuiteSummary:
    suite_id: str
    scenarios: List[ScenarioResult]
    aggregate: Dict[str, Any]
    output_dir: Path


# --------------------------------------------------------------------------- #
# Suite specification helpers
# --------------------------------------------------------------------------- #


def extract_suite_config(
    base_config: Dict[str, Any], suite_override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    cfg = deepcopy(base_config.get("backtest", {}).get("suite", {}) or {})
    if suite_override:
        cfg.update(deepcopy(suite_override))
    return cfg


def _flatten_coin_list(value: Any) -> List[str]:
    if isinstance(value, dict):
        coins = set(value.get("long", [])) | set(value.get("short", []))
        return sorted(coins)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _coerce_exchange_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ValueError("scenario exchanges must be a string or list of strings")


def _coerce_coin_source_dict(value: Any) -> Optional[Dict[str, str]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("scenario coin_sources must be a mapping of coin -> exchange")
    return {str(coin): str(exchange) for coin, exchange in value.items() if exchange is not None}


def resolve_coin_sources(
    base_sources: Dict[str, str],
    overrides: Optional[Dict[str, str]],
) -> Dict[str, str]:
    resolved = dict(base_sources or {})
    if overrides:
        resolved.update(overrides)
    return resolved


def _collect_union(values: Iterable[Optional[List[str]]], fallback: List[str]) -> List[str]:
    union: set[str] = set()
    for val in values:
        if not val:
            continue
        union.update(val)
    if not union:
        union.update(fallback)
    return sorted(union)


def _collect_date_window(
    base_start: str,
    base_end: str,
    scenarios: Sequence[SuiteScenario],
) -> Tuple[str, str]:
    start_dates = [base_start]
    end_dates = [base_end]
    for scenario in scenarios:
        if scenario.start_date:
            start_dates.append(scenario.start_date)
        if scenario.end_date:
            end_dates.append(scenario.end_date)
    return min(start_dates), max(end_dates)


def build_scenarios(
    suite_cfg: Dict[str, Any],
) -> Tuple[List[SuiteScenario], Dict[str, Any], bool, str]:
    scenarios_cfg = suite_cfg.get("scenarios") or []
    if not scenarios_cfg:
        raise ValueError("config.backtest.suite.scenarios must contain at least one scenario.")

    scenarios: List[SuiteScenario] = []
    for idx, raw in enumerate(scenarios_cfg, 1):
        exchanges_value = raw.get("exchanges")
        coin_sources_value = raw.get("coin_sources")
        exchanges_list = _coerce_exchange_list(exchanges_value) if exchanges_value else None
        coin_source_map = (
            _coerce_coin_source_dict(coin_sources_value) if coin_sources_value else None
        )
        scenarios.append(
            SuiteScenario(
                label=str(raw.get("label") or f"scenario_{idx:02d}"),
                start_date=raw.get("start_date"),
                end_date=raw.get("end_date"),
                coins=list(raw.get("coins", [])) if raw.get("coins") is not None else None,
                ignored_coins=(
                    list(raw.get("ignored_coins", []))
                    if raw.get("ignored_coins") is not None
                    else None
                ),
                exchanges=exchanges_list,
                coin_sources=coin_source_map,
            )
        )

    aggregate_cfg = deepcopy(suite_cfg.get("aggregate", {"default": "mean"}))
    include_base = bool(suite_cfg.get("include_base_scenario", False))
    base_label = str(suite_cfg.get("base_label") or "base")
    return scenarios, aggregate_cfg, include_base, base_label


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #


@dataclass
class ExchangeDataset:
    exchange: str
    coins: List[str]
    coin_index: Dict[str, int]
    hlcvs: np.ndarray
    mss: Dict[str, Any]
    btc_usd_prices: np.ndarray
    timestamps: Optional[np.ndarray]
    cache_dir: str


async def prepare_master_datasets(
    base_config: Dict[str, Any],
    exchanges: List[str],
) -> Dict[str, ExchangeDataset]:
    from backtest import prepare_hlcvs_mss

    datasets: Dict[str, ExchangeDataset] = {}
    if require_config_value(base_config, "backtest.combine_ohlcvs"):
        (
            coins,
            hlcvs,
            mss,
            _store_path,
            cache_dir,
            btc_usd_prices,
            timestamps,
        ) = await prepare_hlcvs_mss(base_config, "combined")
        coin_index = {coin: idx for idx, coin in enumerate(coins)}
        datasets["combined"] = ExchangeDataset(
            exchange="combined",
            coins=coins,
            coin_index=coin_index,
            hlcvs=hlcvs,
            mss=mss,
            btc_usd_prices=btc_usd_prices,
            timestamps=timestamps,
            cache_dir=str(cache_dir),
        )
    else:
        for exchange in exchanges:
            (
                coins,
                hlcvs,
                mss,
                _store_path,
                cache_dir,
                btc_usd_prices,
                timestamps,
            ) = await prepare_hlcvs_mss(base_config, exchange)
            coin_index = {coin: idx for idx, coin in enumerate(coins)}
            datasets[exchange] = ExchangeDataset(
                exchange=exchange,
                coins=coins,
                coin_index=coin_index,
                hlcvs=hlcvs,
                mss=mss,
                btc_usd_prices=btc_usd_prices,
                timestamps=timestamps,
                cache_dir=str(cache_dir),
            )
    return datasets


# --------------------------------------------------------------------------- #
# Scenario execution
# --------------------------------------------------------------------------- #


def apply_scenario(
    base_config: Dict[str, Any],
    scenario: SuiteScenario,
    master_coins: List[str],
    master_ignored: List[str],
    available_exchanges: Iterable[str],
    available_coins: set[str],
) -> Tuple[Dict[str, Any], List[str]]:
    cfg = deepcopy(base_config)
    cfg["backtest"]["start_date"] = scenario.start_date or cfg["backtest"]["start_date"]
    cfg["backtest"]["end_date"] = scenario.end_date or cfg["backtest"]["end_date"]

    scenario_coins = list(scenario.coins) if scenario.coins is not None else list(master_coins)
    scenario_ignored = (
        list(scenario.ignored_coins) if scenario.ignored_coins is not None else list(master_ignored)
    )

    filtered_coins = [coin for coin in scenario_coins if coin in available_coins]
    missing = sorted(set(scenario_coins) - set(filtered_coins))
    if missing:
        logging.warning(
            "Scenario %s: skipping %d coin(s) missing in dataset: %s",
            scenario.label,
            len(missing),
            ",".join(missing),
        )
    if not filtered_coins:
        raise ValueError(f"Scenario {scenario.label} has no usable coins.")
    filtered_coins = sorted(dict.fromkeys(filtered_coins))

    filtered_ignored = [coin for coin in scenario_ignored if coin in available_coins]
    filtered_ignored = sorted(dict.fromkeys(filtered_ignored))

    cfg.setdefault("backtest", {}).setdefault("coins", {})
    cfg.setdefault("backtest", {}).setdefault("cache_dir", {})
    cfg.setdefault("live", {}).setdefault("approved_coins", {})
    cfg.setdefault("live", {}).setdefault("ignored_coins", {})

    for exchange in available_exchanges:
        cfg["backtest"]["coins"][exchange] = filtered_coins

    if isinstance(cfg["live"]["approved_coins"], dict):
        cfg["live"]["approved_coins"]["long"] = list(filtered_coins)
        cfg["live"]["approved_coins"]["short"] = list(filtered_coins)
    else:
        cfg["live"]["approved_coins"] = list(filtered_coins)

    if isinstance(cfg["live"]["ignored_coins"], dict):
        cfg["live"]["ignored_coins"]["long"] = list(filtered_ignored)
        cfg["live"]["ignored_coins"]["short"] = list(filtered_ignored)
    else:
        cfg["live"]["ignored_coins"] = list(filtered_ignored)

    return cfg, filtered_coins


async def run_backtest_scenario(
    scenario: SuiteScenario,
    base_config: Dict[str, Any],
    datasets: Dict[str, ExchangeDataset],
    master_coins: List[str],
    master_ignored: List[str],
    available_coins: set[str],
    results_root: Optional[Path],
    disable_plotting: bool,
) -> ScenarioResult:
    from backtest import run_backtest, post_process

    scenario_config, scenario_coins = apply_scenario(
        base_config,
        scenario,
        master_coins=master_coins,
        master_ignored=master_ignored,
        available_exchanges=[ds.exchange for ds in datasets.values()],
        available_coins=available_coins,
    )
    scenario_config["disable_plotting"] = disable_plotting

    per_exchange: Dict[str, Dict[str, Any]] = {}
    start_ts = utc_ms()
    scenario_dir = None
    if results_root is not None:
        scenario_dir = results_root / scenario.label
        scenario_dir.mkdir(parents=True, exist_ok=True)

    for exchange_key, dataset in datasets.items():
        exchange_coins = [coin for coin in scenario_coins if coin in dataset.coin_index]
        if not exchange_coins:
            logging.warning(
                "Scenario %s: no overlapping coins for exchange %s; skipping.",
                scenario.label,
                dataset.exchange,
            )
            continue
        scenario_config["backtest"]["coins"][dataset.exchange] = exchange_coins
        scenario_config["backtest"]["cache_dir"][dataset.exchange] = dataset.cache_dir

        indices = [dataset.coin_index[coin] for coin in exchange_coins]
        hlcvs_slice = dataset.hlcvs[:, indices, :]
        mss_slice = {coin: dataset.mss.get(coin, {}) for coin in exchange_coins}
        if "__meta__" in dataset.mss:
            mss_slice["__meta__"] = dataset.mss["__meta__"]

        fills, equities_array, analysis = run_backtest(
            hlcvs_slice,
            mss_slice,
            scenario_config,
            dataset.exchange,
            dataset.btc_usd_prices,
            dataset.timestamps,
        )

        per_exchange[exchange_key] = analysis
        if scenario_dir is not None:
            try:
                post_process(
                    scenario_config,
                    hlcvs_slice,
                    fills,
                    equities_array,
                    dataset.btc_usd_prices,
                    analysis,
                    str(scenario_dir / dataset.exchange),
                    dataset.exchange,
                )
            except Exception as exc:
                logging.error(
                    "Scenario %s exchange %s: post-process failed (%s)",
                    scenario.label,
                    dataset.exchange,
                    exc,
                )
        del fills
        del equities_array

    if not per_exchange:
        raise ValueError(f"Scenario {scenario.label} had no exchanges after filtering.")

    from tools.iterative_backtester import combine_analyses

    combined_metrics = combine_analyses(per_exchange)
    elapsed = (utc_ms() - start_ts) / 1000.0
    return ScenarioResult(
        scenario=scenario,
        per_exchange=per_exchange,
        combined_metrics=combined_metrics,
        elapsed_seconds=elapsed,
        output_path=scenario_dir,
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate_metrics(
    results: Sequence[ScenarioResult], aggregate_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    if not results:
        return {"aggregated": {}, "stats": {}}
    metrics_values: Dict[str, List[float]] = {}
    for result in results:
        for key, value in result.combined_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics_values.setdefault(key, []).append(float(value))

    stats: Dict[str, Dict[str, float]] = {}
    aggregates: Dict[str, float] = {}
    default_mode = str(aggregate_cfg.get("default", "mean")).lower()

    for metric, values in metrics_values.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        stats[metric] = {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }
        mode = aggregate_cfg.get(metric)
        if mode is None and "_" in metric:
            base = metric.rsplit("_", 1)[0]
            mode = aggregate_cfg.get(base)
        mode = str(mode or default_mode).lower()
        if mode == "mean":
            aggregates[metric] = stats[metric]["mean"]
        elif mode == "max":
            aggregates[metric] = stats[metric]["max"]
        elif mode == "min":
            aggregates[metric] = stats[metric]["min"]
        elif mode == "std":
            aggregates[metric] = stats[metric]["std"]
        elif mode == "median":
            aggregates[metric] = float(np.median(arr))
        else:
            raise ValueError(f"Unsupported aggregation mode '{mode}' for metric '{metric}'.")
    return {"aggregated": aggregates, "stats": stats}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


async def run_backtest_suite_async(
    config: Dict[str, Any],
    suite_cfg: Dict[str, Any],
    *,
    disable_plotting: bool,
    suite_output_root: Optional[Path] = None,
) -> SuiteSummary:
    manage_rust_compilation()

    exchanges_list = require_config_value(config, "backtest.exchanges")
    for exchange in exchanges_list:
        await load_markets(exchange, verbose=False)
    await format_approved_ignored_coins(config, exchanges_list, verbose=False)

    base_start = require_config_value(config, "backtest.start_date")
    base_end = require_config_value(config, "backtest.end_date")
    base_coins = _flatten_coin_list(require_live_value(config, "approved_coins"))
    base_ignored = _flatten_coin_list(require_live_value(config, "ignored_coins"))

    scenarios, aggregate_cfg, include_base, base_label = build_scenarios(suite_cfg)
    if include_base:
        base_scenario = SuiteScenario(
            label=base_label,
            start_date=base_start,
            end_date=base_end,
            coins=list(base_coins),
            ignored_coins=list(base_ignored),
        )
        scenarios = [base_scenario] + list(scenarios)

    master_coins = _collect_union((s.coins for s in scenarios), base_coins)
    master_ignored = _collect_union((s.ignored_coins for s in scenarios), base_ignored)
    global_start, global_end = _collect_date_window(base_start, base_end, scenarios)

    base_config = deepcopy(config)
    base_config["backtest"]["start_date"] = global_start
    base_config["backtest"]["end_date"] = global_end
    if isinstance(base_config["live"]["approved_coins"], dict):
        base_config["live"]["approved_coins"]["long"] = list(master_coins)
        base_config["live"]["approved_coins"]["short"] = list(master_coins)
    else:
        base_config["live"]["approved_coins"] = list(master_coins)
    if isinstance(base_config["live"]["ignored_coins"], dict):
        base_config["live"]["ignored_coins"]["long"] = list(master_ignored)
        base_config["live"]["ignored_coins"]["short"] = list(master_ignored)
    else:
        base_config["live"]["ignored_coins"] = list(master_ignored)

    datasets = await prepare_master_datasets(base_config, exchanges_list)
    available_coins: set[str] = set()
    for dataset in datasets.values():
        available_coins.update(dataset.coins)

    suite_timestamp = ts_to_date(utc_ms())[:19].replace(":", "_")
    suite_dir = (
        suite_output_root
        if suite_output_root is not None
        else Path(require_config_value(config, "backtest.base_dir")) / "suite_runs" / suite_timestamp
    )
    suite_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting backtest suite: %d scenario(s)", len(scenarios))
    results: List[ScenarioResult] = []
    for scenario in scenarios:
        logging.info("Running scenario '%s'...", scenario.label)
        result = await run_backtest_scenario(
            scenario,
            base_config,
            datasets,
            master_coins,
            master_ignored,
            available_coins,
            suite_dir,
            disable_plotting=disable_plotting,
        )
        results.append(result)
        logging.info(
            "Scenario %s finished in %.2fs with %d metrics.",
            scenario.label,
            result.elapsed_seconds,
            len(result.combined_metrics),
        )

    aggregate_summary = aggregate_metrics(results, aggregate_cfg)
    summary_payload = {
        "suite_id": suite_timestamp,
        "meta": {
            "scenarios": [res.scenario.label for res in results],
            "timestamp": suite_timestamp,
        },
        "aggregate": aggregate_summary,
        "per_scenario": {
            res.scenario.label: {
                "metrics": res.combined_metrics,
                "elapsed_seconds": res.elapsed_seconds,
            }
            for res in results
        },
    }
    (suite_dir / "suite_summary.json").write_text(json.dumps(summary_payload, indent=2))

    return SuiteSummary(
        suite_id=suite_timestamp,
        scenarios=results,
        aggregate=aggregate_summary,
        output_dir=suite_dir,
    )


def run_backtest_suite_sync(
    config_path: Path,
    *,
    suite_config_path: Optional[Path] = None,
    disable_plotting: bool = False,
) -> SuiteSummary:
    configure_logging()
    config = load_config(str(config_path), verbose=False)
    config = format_config(config, verbose=False)
    config = parse_overrides(config, verbose=False)

    suite_override = None
    if suite_config_path:
        suite_override = (
            load_config(str(suite_config_path), verbose=False).get("backtest", {}).get("suite")
        )
        if suite_override is None:
            raise ValueError(
                f"Suite config {suite_config_path} does not contain backtest.suite definition."
            )

    suite_cfg = extract_suite_config(config, suite_override)
    if not suite_cfg.get("scenarios"):
        raise ValueError("Suite configuration must define at least one scenario.")

    return asyncio.run(
        run_backtest_suite_async(
            config,
            suite_cfg,
            disable_plotting=disable_plotting,
        )
    )


# --------------------------------------------------------------------------- #
# Legacy compatibility shim
# --------------------------------------------------------------------------- #


def cli_entrypoint(config_path: str, suite_config_path: Optional[str] = None) -> None:
    summary = run_backtest_suite_sync(
        Path(config_path),
        suite_config_path=Path(suite_config_path) if suite_config_path else None,
    )
    logging.info(
        "Suite %s completed | scenarios=%d | output=%s",
        summary.suite_id,
        len(summary.scenarios),
        summary.output_dir,
    )
