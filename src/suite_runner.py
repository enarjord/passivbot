"""
Shared helpers for running backtest/optimizer suites.

The suite runner prepares shared datasets, applies scenario overrides, and
invokes the existing backtest pipeline for every scenario.  Both the CLI
backtester and the optimizer import this module when operating in suite mode.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
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
from config_transform import ConfigTransformTracker, record_transform
from logging_setup import configure_logging
from utils import (
    format_approved_ignored_coins,
    format_end_date,
    load_markets,
    symbol_to_coin,
    ts_to_date,
    utc_ms,
    date_to_ts,
)
from downloader import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes
from ohlcv_utils import align_and_aggregate_hlcvs
from shared_arrays import SharedArraySpec
from metrics_schema import flatten_metric_stats, merge_suite_payload


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
    overrides: Optional[Dict[str, Any]] = None


@dataclass
class ScenarioResult:
    scenario: SuiteScenario
    per_exchange: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]
    elapsed_seconds: float
    output_path: Optional[Path]


@dataclass
class SuiteSummary:
    suite_id: str
    scenarios: List[ScenarioResult]
    aggregate: Dict[str, Any]
    output_dir: Path
    suite_metrics: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Suite specification helpers
# --------------------------------------------------------------------------- #


def extract_suite_config(
    base_config: Dict[str, Any], suite_override: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract suite configuration from the new flattened config structure.

    New structure reads from:
    - backtest.scenarios (list of scenario dicts)
    - backtest.aggregate (aggregation settings)
    - backtest.exchanges (default exchanges for scenarios)
    - backtest.volume_normalization (bool, default True)
    - backtest.suite_enabled (bool, default True) - master switch for suite mode

    Args:
        base_config: Full config dict
        suite_override: Optional override dict with scenarios/aggregate keys

    Returns:
        Dict with 'scenarios', 'aggregate', 'exchanges', 'volume_normalization', and 'enabled' keys
    """
    backtest = base_config.get("backtest", {})

    # Build config from new flattened structure
    cfg = {
        "scenarios": deepcopy(backtest.get("scenarios", [])),
        "aggregate": deepcopy(backtest.get("aggregate", {"default": "mean"})),
        "exchanges": deepcopy(backtest.get("exchanges", [])),
        "volume_normalization": backtest.get("volume_normalization", True),
    }

    # Apply overrides if provided
    if suite_override:
        if "scenarios" in suite_override:
            cfg["scenarios"] = deepcopy(suite_override["scenarios"])
        if "aggregate" in suite_override:
            cfg["aggregate"] = deepcopy(suite_override["aggregate"])
        if "exchanges" in suite_override:
            cfg["exchanges"] = deepcopy(suite_override["exchanges"])
        if "volume_normalization" in suite_override:
            cfg["volume_normalization"] = suite_override["volume_normalization"]

    # Determine if suite mode is enabled:
    # - suite_enabled config param must be true (default: true)
    # - AND scenarios must exist
    suite_enabled_config = backtest.get("suite_enabled", True)
    has_scenarios = bool(cfg.get("scenarios"))
    cfg["enabled"] = suite_enabled_config and has_scenarios

    return cfg


def filter_scenarios_by_label(
    scenarios: List[Dict[str, Any]],
    labels: List[str],
) -> List[Dict[str, Any]]:
    """Filter scenarios to only include those matching the given labels.

    Args:
        scenarios: List of scenario dicts (each with a 'label' key)
        labels: List of labels to keep

    Returns:
        Filtered list of scenarios

    Raises:
        ValueError: If no scenarios match the given labels
    """
    if not labels:
        return scenarios

    label_set = set(labels)
    filtered = [s for s in scenarios if s.get("label") in label_set]

    if not filtered:
        available = [s.get("label", f"<unnamed_{i}>") for i, s in enumerate(scenarios)]
        raise ValueError(
            f"No scenarios match the requested labels {labels}. "
            f"Available labels: {available}"
        )

    return filtered


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
    normalized: Dict[str, str] = {}
    raw_keys: Dict[str, str] = {}
    for coin, exchange in value.items():
        if exchange is None:
            continue
        coin_key = symbol_to_coin(str(coin), verbose=False)
        if not coin_key:
            continue
        exchange_value = str(exchange)
        existing = normalized.get(coin_key)
        if existing is not None and existing != exchange_value:
            raise ValueError(
                "scenario coin_sources maps conflicting exchanges for "
                f"{coin_key}: {raw_keys.get(coin_key, coin_key)}->{existing} and {coin}->{exchange_value}"
            )
        normalized[coin_key] = exchange_value
        raw_keys.setdefault(coin_key, str(coin))
    return normalized or None


def _normalize_coin_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    coins_raw = _flatten_coin_list(value)
    normalized: List[str] = []
    seen: set[str] = set()
    for entry in coins_raw:
        coin = symbol_to_coin(str(entry), verbose=False)
        if not coin or coin in seen:
            continue
        seen.add(coin)
        normalized.append(coin)
    return normalized


def resolve_coin_sources(
    base_sources: Optional[Dict[str, str]],
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
    base_exchanges: Optional[List[str]] = None,
) -> Tuple[List[SuiteScenario], Dict[str, Any]]:
    """Build list of SuiteScenario objects from suite config.

    In the new flattened structure:
    - Scenarios without explicit 'exchanges' inherit from suite_cfg['exchanges'] or base_exchanges
    - Single exchange in scenario = use that exchange's data
    - Multiple exchanges in scenario = best-per-coin combination

    Args:
        suite_cfg: Suite configuration dict with 'scenarios' and 'aggregate'
        base_exchanges: Default exchanges to inherit when scenario doesn't specify

    Returns:
        Tuple of (scenarios list, aggregate config dict)
    """
    scenarios_cfg = suite_cfg.get("scenarios") or []
    if not scenarios_cfg:
        raise ValueError("config.backtest.scenarios must contain at least one scenario.")

    default_exchanges = suite_cfg.get("exchanges") or base_exchanges or []

    scenarios: List[SuiteScenario] = []
    for idx, raw in enumerate(scenarios_cfg, 1):
        exchanges_value = raw.get("exchanges")
        coin_sources_value = raw.get("coin_sources")

        # Resolve exchanges: scenario-specific or inherit from defaults
        if exchanges_value:
            exchanges_list = _coerce_exchange_list(exchanges_value)
        elif default_exchanges:
            exchanges_list = list(default_exchanges)
        else:
            exchanges_list = None

        coin_source_map = _coerce_coin_source_dict(coin_sources_value) if coin_sources_value else None
        overrides = raw.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError(f"Scenario overrides for '{raw.get('label')}' must be a mapping")
        scenario_coins = (
            _normalize_coin_list(raw.get("coins")) if raw.get("coins") is not None else None
        )
        scenario_ignored = (
            _normalize_coin_list(raw.get("ignored_coins"))
            if raw.get("ignored_coins") is not None
            else None
        )
        scenarios.append(
            SuiteScenario(
                label=str(raw.get("label") or f"scenario_{idx:02d}"),
                start_date=raw.get("start_date"),
                end_date=raw.get("end_date"),
                coins=scenario_coins,
                ignored_coins=scenario_ignored,
                exchanges=exchanges_list,
                coin_sources=coin_source_map,
                overrides=deepcopy(overrides) if overrides else None,
            )
        )

    aggregate_cfg = deepcopy(suite_cfg.get("aggregate", {"default": "mean"}))
    return scenarios, aggregate_cfg


def collect_suite_coin_sources(
    config: Dict[str, Any],
    scenarios: Sequence[SuiteScenario],
) -> Dict[str, str]:
    """
    Merge baseline coin_sources with any scenario overrides.

    The merged mapping is shared across the suite so all scenarios consume a
    consistent view of the underlying exchange assignment.  Conflicting
    requests raise immediately to avoid silently running scenarios with
    mismatched data.
    """

    base_sources = deepcopy(config.get("backtest", {}).get("coin_sources") or {})
    merged: Dict[str, str] = {
        str(coin): str(exchange) for coin, exchange in base_sources.items() if exchange is not None
    }
    for scenario in scenarios:
        if not scenario.coin_sources:
            continue
        for coin, exchange in scenario.coin_sources.items():
            if exchange is None:
                continue
            coin_key = str(coin)
            exchange_value = str(exchange)
            existing = merged.get(coin_key)
            if existing and existing != exchange_value:
                raise ValueError(
                    f"Scenario '{scenario.label}' forces {coin_key} to exchange "
                    f"{exchange_value}, but another scenario already forces {coin_key} "
                    f"to {existing}. Please align coin_sources across the suite."
                )
            merged[coin_key] = exchange_value
    return merged


def filter_coins_by_exchange_assignment(
    coins: Sequence[str],
    allowed_exchanges: Optional[Sequence[str]],
    coin_exchange_map: Dict[str, str],
    *,
    default_exchange: str,
) -> Tuple[List[str], List[str]]:
    """
    Split the provided coins into those whose assigned exchange is allowed and
    those that should be skipped.
    """

    allowed_set = {str(ex) for ex in allowed_exchanges} if allowed_exchanges else None
    selected: List[str] = []
    skipped: List[str] = []
    for coin in coins:
        assigned_exchange = coin_exchange_map.get(coin, default_exchange)
        if allowed_set and assigned_exchange not in allowed_set:
            skipped.append(coin)
            continue
        selected.append(coin)
    return selected, skipped


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #


@dataclass
@dataclass
class ExchangeDataset:
    exchange: str
    coins: List[str]
    coin_index: Dict[str, int]
    coin_exchange: Dict[str, str]
    available_exchanges: List[str]
    hlcvs: np.ndarray
    mss: Dict[str, Any]
    btc_usd_prices: np.ndarray
    timestamps: Optional[np.ndarray]
    cache_dir: str
    hlcvs_spec: Optional[SharedArraySpec] = None
    btc_spec: Optional[SharedArraySpec] = None
    hlcvs_spec: Optional[SharedArraySpec] = None
    btc_spec: Optional[SharedArraySpec] = None


def _determine_needed_individual_exchanges(
    scenarios: List["SuiteScenario"],
    base_exchanges: List[str],
) -> Set[str]:
    """
    Analyze scenarios to determine which individual exchange datasets are needed.

    Returns a set of exchange names that need individual datasets (for scenarios
    that restrict to a subset of exchanges). Empty set means only combined is needed.
    """
    base_set = set(base_exchanges)
    needed: Set[str] = set()

    for scenario in scenarios:
        if scenario.exchanges:
            scenario_set = set(scenario.exchanges)
            # If scenario uses a strict subset, we need individual datasets for those exchanges
            if scenario_set and scenario_set != base_set:
                needed.update(scenario_set)

    return needed


def _apply_candle_aggregation(hlcvs, timestamps, btc_usd_prices, mss, interval):
    """Aggregate candles and update mss metadata. Returns (hlcvs, timestamps, btc_usd_prices)."""
    n_before = hlcvs.shape[0]
    hlcvs, timestamps, btc_usd_prices, offset_bars = align_and_aggregate_hlcvs(
        hlcvs, timestamps, btc_usd_prices, int(interval)
    )
    logging.debug(
        "[suite] aggregated %dm candles: %d bars -> %d bars (trimmed %d for alignment)",
        interval, n_before, hlcvs.shape[0], offset_bars,
    )
    meta = mss.setdefault("__meta__", {})
    meta["data_interval_minutes"] = int(interval)
    meta["candle_interval_offset_bars"] = int(offset_bars)
    return hlcvs, timestamps, btc_usd_prices


async def prepare_master_datasets(
    base_config: Dict[str, Any],
    exchanges: List[str],
    shared_array_manager=None,
    *,
    needed_individual_exchanges: Optional[Set[str]] = None,
    candle_interval_minutes: int = 1,
) -> Dict[str, ExchangeDataset]:
    from backtest import prepare_hlcvs_mss

    datasets: Dict[str, ExchangeDataset] = {}

    def _build_dataset(
        exchange_label: str,
        exchange_name: str,
        coins: List[str],
        hlcvs: np.ndarray,
        mss: Dict[str, Any],
        cache_dir: str,
        btc_usd_prices: np.ndarray,
        timestamps: Optional[np.ndarray],
    ) -> ExchangeDataset:
        coin_index = {coin: idx for idx, coin in enumerate(coins)}
        coin_exchange = {
            coin: str(mss.get(coin, {}).get("exchange", exchange_name)) for coin in coins
        }
        available_exchanges = sorted(set(coin_exchange.values())) or [exchange_name]
        hlcvs_array = np.ascontiguousarray(hlcvs, dtype=np.float64)
        btc_array = np.ascontiguousarray(btc_usd_prices, dtype=np.float64)
        hlcvs_spec = None
        btc_spec = None
        if shared_array_manager is not None:
            # Copy to SharedMemory, then reassign to view (frees intermediate copy)
            hlcvs_spec, hlcvs_view = shared_array_manager.create_from(hlcvs_array)
            del hlcvs_array  # Free intermediate contiguous array
            hlcvs_array = hlcvs_view
            btc_spec, btc_view = shared_array_manager.create_from(btc_array)
            del btc_array  # Free intermediate contiguous array
            btc_array = btc_view
        return ExchangeDataset(
            exchange=exchange_label,
            coins=coins,
            coin_index=coin_index,
            coin_exchange=coin_exchange,
            available_exchanges=available_exchanges,
            hlcvs=hlcvs_array,
            mss=mss,
            btc_usd_prices=btc_array,
            timestamps=timestamps,
            cache_dir=str(cache_dir),
            hlcvs_spec=hlcvs_spec,
            btc_spec=btc_spec,
        )

    # Data strategy:
    # - Single exchange = use that exchange's data only
    # - Multiple exchanges = prepare combined (best-per-coin) dataset
    # - If any scenario restricts to a subset of exchanges, also prepare
    #   individual datasets for those exchanges (determined by caller)
    use_combined = len(exchanges) > 1

    if use_combined:
        # Prepare combined (best-per-coin) dataset
        (
            coins,
            hlcvs,
            mss,
            _store_path,
            cache_dir,
            btc_usd_prices,
            timestamps,
        ) = await prepare_hlcvs_mss(base_config, "combined")
        if candle_interval_minutes > 1:
            hlcvs, timestamps, btc_usd_prices = _apply_candle_aggregation(
                hlcvs, timestamps, btc_usd_prices, mss, candle_interval_minutes
            )
        datasets["combined"] = _build_dataset(
            "combined",
            "combined",
            coins,
            hlcvs,
            mss,
            cache_dir,
            btc_usd_prices,
            timestamps,
        )
        # Free original arrays after copying to SharedMemory (can save ~5GB+ RAM)
        del hlcvs, btc_usd_prices

        # Only prepare individual exchange datasets if scenarios need them
        if needed_individual_exchanges:
            for exchange in exchanges:
                if exchange not in needed_individual_exchanges:
                    continue
                logging.info(
                    "Preparing individual %s dataset for single-exchange scenarios",
                    exchange,
                )
                (
                    ex_coins,
                    ex_hlcvs,
                    ex_mss,
                    _ex_store_path,
                    ex_cache_dir,
                    ex_btc_usd_prices,
                    ex_timestamps,
                ) = await prepare_hlcvs_mss(base_config, exchange)
                if candle_interval_minutes > 1:
                    ex_hlcvs, ex_timestamps, ex_btc_usd_prices = _apply_candle_aggregation(
                        ex_hlcvs, ex_timestamps, ex_btc_usd_prices, ex_mss, candle_interval_minutes
                    )
                datasets[exchange] = _build_dataset(
                    exchange,
                    exchange,
                    ex_coins,
                    ex_hlcvs,
                    ex_mss,
                    ex_cache_dir,
                    ex_btc_usd_prices,
                    ex_timestamps,
                )
                # Free original arrays after copying to SharedMemory
                del ex_hlcvs, ex_btc_usd_prices
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
            if candle_interval_minutes > 1:
                hlcvs, timestamps, btc_usd_prices = _apply_candle_aggregation(
                    hlcvs, timestamps, btc_usd_prices, mss, candle_interval_minutes
                )
            datasets[exchange] = _build_dataset(
                exchange,
                exchange,
                coins,
                hlcvs,
                mss,
                cache_dir,
                btc_usd_prices,
                timestamps,
            )
            # Free original arrays after copying to SharedMemory
            del hlcvs, btc_usd_prices
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
    base_coin_sources: Optional[Dict[str, str]] = None,
    *,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    cfg = deepcopy(base_config)
    tracker = ConfigTransformTracker()
    backtest_section = cfg.setdefault("backtest", {})
    live_section = cfg.setdefault("live", {})

    new_start = scenario.start_date or backtest_section.get("start_date")
    if new_start != backtest_section.get("start_date"):
        tracker.update(["backtest", "start_date"], backtest_section.get("start_date"), new_start)
        backtest_section["start_date"] = new_start

    new_end = scenario.end_date or backtest_section.get("end_date")
    if new_end != backtest_section.get("end_date"):
        tracker.update(["backtest", "end_date"], backtest_section.get("end_date"), new_end)
        backtest_section["end_date"] = new_end

    scenario_coins = list(scenario.coins) if scenario.coins is not None else list(master_coins)
    scenario_ignored = (
        list(scenario.ignored_coins) if scenario.ignored_coins is not None else list(master_ignored)
    )

    filtered_coins = [coin for coin in scenario_coins if coin in available_coins]
    missing = sorted(set(scenario_coins) - set(filtered_coins))
    if missing and not quiet:
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

    scenario_exchanges = list(scenario.exchanges) if scenario.exchanges else list(available_exchanges)
    if scenario_exchanges != backtest_section.get("exchanges"):
        tracker.update(
            ["backtest", "exchanges"], backtest_section.get("exchanges"), scenario_exchanges
        )
        backtest_section["exchanges"] = scenario_exchanges
    backtest_section.setdefault("coins", {})
    backtest_section.setdefault("cache_dir", {})
    live_section.setdefault("approved_coins", {})
    live_section.setdefault("ignored_coins", {})

    for exchange in scenario_exchanges:
        current = backtest_section["coins"].get(exchange)
        if current != filtered_coins:
            tracker.update(
                ["backtest", "coins", exchange],
                deepcopy(current),
                list(filtered_coins),
            )
        backtest_section["coins"][exchange] = list(filtered_coins)

    if isinstance(live_section["approved_coins"], dict):
        for side in ("long", "short"):
            current = live_section["approved_coins"].get(side, [])
            if current != filtered_coins:
                tracker.update(
                    ["live", "approved_coins", side],
                    deepcopy(current),
                    list(filtered_coins),
                )
            live_section["approved_coins"][side] = list(filtered_coins)
    else:
        current = live_section.get("approved_coins")
        if current != filtered_coins:
            tracker.update(["live", "approved_coins"], deepcopy(current), list(filtered_coins))
        live_section["approved_coins"] = list(filtered_coins)

    if isinstance(live_section["ignored_coins"], dict):
        for side in ("long", "short"):
            current = live_section["ignored_coins"].get(side, [])
            if current != filtered_ignored:
                tracker.update(
                    ["live", "ignored_coins", side],
                    deepcopy(current),
                    list(filtered_ignored),
                )
            live_section["ignored_coins"][side] = list(filtered_ignored)
    else:
        current = live_section.get("ignored_coins")
        if current != filtered_ignored:
            tracker.update(["live", "ignored_coins"], deepcopy(current), list(filtered_ignored))
        live_section["ignored_coins"] = list(filtered_ignored)

    resolved_sources = resolve_coin_sources(
        base_coin_sources or {},
        scenario.coin_sources,
    )
    if resolved_sources != backtest_section.get("coin_sources"):
        tracker.update(
            ["backtest", "coin_sources"],
            deepcopy(backtest_section.get("coin_sources")),
            deepcopy(resolved_sources),
        )
        backtest_section["coin_sources"] = resolved_sources

    if scenario.overrides:
        for dotted_path, value in scenario.overrides.items():
            if not isinstance(dotted_path, str):
                raise ValueError(f"Scenario '{scenario.label}' override keys must be dotted strings")
            _apply_override(cfg, dotted_path, value, tracker)

    if tracker.summary() and not quiet:
        details = tracker.merge_details({"scenario": scenario.label})
        record_transform(cfg, "apply_scenario", details)

    return cfg, filtered_coins


def _normalize_coins_by_exchange(coin_exchange: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for coin, exchange in coin_exchange.items():
        grouped.setdefault(str(exchange), []).append(str(coin))
    return {ex: sorted(coins) for ex, coins in sorted(grouped.items())}


def _compute_effective_coin_exchange(
    scenario: SuiteScenario,
    scenario_coins: List[str],
    datasets: Dict[str, "ExchangeDataset"],
    available_exchanges: List[str],
) -> Dict[str, str]:
    """Return effective coin->exchange assignment for a scenario."""
    has_combined = "combined" in datasets
    raw_scenario_exchanges = set(scenario.exchanges) if scenario.exchanges else None
    actual_exchanges_set = {ex for ex in available_exchanges if ex != "combined"}

    if raw_scenario_exchanges:
        scenario_exchanges = raw_scenario_exchanges & actual_exchanges_set
        if not scenario_exchanges:
            scenario_exchanges = actual_exchanges_set
    else:
        scenario_exchanges = actual_exchanges_set

    use_combined = has_combined and scenario_exchanges == actual_exchanges_set
    coin_exchange: Dict[str, str] = {}

    if use_combined:
        dataset = datasets["combined"]
        allowed_exchanges = (
            list(scenario.exchanges) if scenario.exchanges else list(dataset.available_exchanges)
        )
        selected_coins, _ = filter_coins_by_exchange_assignment(
            scenario_coins,
            allowed_exchanges,
            dataset.coin_exchange,
            default_exchange=dataset.exchange,
        )
        for coin in selected_coins:
            coin_exchange[coin] = dataset.coin_exchange.get(coin, dataset.exchange)
    else:
        for key, dataset in datasets.items():
            if key == "combined":
                continue
            if dataset.exchange not in scenario_exchanges and key not in scenario_exchanges:
                continue
            coins_for_exchange = [coin for coin in scenario_coins if coin in dataset.coin_index]
            for coin in coins_for_exchange:
                coin_exchange[coin] = dataset.coin_exchange.get(coin, dataset.exchange)
    return coin_exchange


def _build_scenario_signature(
    scenario_config: Dict[str, Any],
    coin_exchange: Dict[str, str],
) -> str:
    """Build a stable signature for scenario deduplication."""
    payload = deepcopy(scenario_config)
    # Ignore transform metadata; it differs per scenario but doesn't affect results.
    payload.pop("_transform_log", None)
    backtest_section = payload.setdefault("backtest", {})
    coins_by_ex = _normalize_coins_by_exchange(coin_exchange)
    backtest_section["coins"] = coins_by_ex
    backtest_section["exchanges"] = sorted(coins_by_ex.keys())
    # cache_dir paths are environment-specific and don't affect results
    backtest_section["cache_dir"] = {}
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _apply_override(
    config: Dict[str, Any], dotted_path: str, value: Any, tracker: ConfigTransformTracker
) -> None:
    parts = dotted_path.split(".")
    if not parts:
        raise ValueError("Override paths must not be empty")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    final_key = parts[-1]
    previous = target.get(final_key)
    if previous != value:
        tracker.update(parts, deepcopy(previous), deepcopy(value))
    target[final_key] = value


async def run_backtest_scenario(
    scenario: SuiteScenario,
    base_config: Dict[str, Any],
    datasets: Dict[str, ExchangeDataset],
    master_coins: List[str],
    master_ignored: List[str],
    available_exchanges: List[str],
    available_coins: set[str],
    results_root: Optional[Path],
    disable_plotting: bool,
    base_coin_sources: Optional[Dict[str, str]] = None,
) -> ScenarioResult:
    from backtest import (
        build_backtest_payload,
        execute_backtest,
        post_process,
    )

    scenario_config, scenario_coins = apply_scenario(
        base_config,
        scenario,
        master_coins=master_coins,
        master_ignored=master_ignored,
        available_exchanges=available_exchanges,
        available_coins=available_coins,
        base_coin_sources=base_coin_sources,
    )
    scenario_config["disable_plotting"] = disable_plotting

    per_exchange: Dict[str, Dict[str, Any]] = {}
    start_ts = utc_ms()
    scenario_dir = None
    if results_root is not None:
        scenario_dir = results_root / scenario.label
        scenario_dir.mkdir(parents=True, exist_ok=True)

    # Determine which dataset(s) to use based on scenario's exchange restriction
    has_combined = "combined" in datasets
    raw_scenario_exchanges = set(scenario.exchanges) if scenario.exchanges else None
    # Compute actual exchanges (excluding "combined" which is a synthetic dataset label)
    actual_exchanges_set = {ex for ex in available_exchanges if ex != "combined"}

    # Filter scenario exchanges to only those actually available in the dataset
    if raw_scenario_exchanges:
        unavailable = raw_scenario_exchanges - actual_exchanges_set
        if unavailable:
            logging.debug(
                "Scenario %s: exchanges %s not available in dataset, using %s",
                scenario.label,
                sorted(unavailable),
                sorted(raw_scenario_exchanges & actual_exchanges_set) or "all available",
            )
        scenario_exchanges = raw_scenario_exchanges & actual_exchanges_set
        if not scenario_exchanges:
            # If no overlap, fall back to all available exchanges
            scenario_exchanges = actual_exchanges_set
    else:
        scenario_exchanges = actual_exchanges_set

    # Use combined dataset when:
    # 1. It exists, AND
    # 2. Scenario uses all actual exchanges (not a strict subset)
    use_combined = has_combined and scenario_exchanges == actual_exchanges_set

    if use_combined:
        dataset = datasets["combined"]
        per_exchange = _run_combined_dataset(
            dataset,
            scenario,
            scenario_config,
            scenario_coins,
            scenario_dir,
            build_backtest_payload,
            execute_backtest,
            post_process,
        )
    else:
        # Use per-exchange datasets for scenarios with exchange restrictions
        # Filter datasets to only include those requested by the scenario
        filtered_datasets = {
            k: v for k, v in datasets.items()
            if k != "combined" and k in scenario_exchanges
        }
        if not filtered_datasets:
            raise ValueError(
                f"Scenario {scenario.label} requests exchanges {scenario_exchanges} "
                f"but no matching datasets are available (have: {list(datasets.keys())})"
            )
        per_exchange = _run_multi_dataset(
            filtered_datasets,
            scenario,
            scenario_config,
            scenario_coins,
            scenario_dir,
            build_backtest_payload,
            execute_backtest,
            post_process,
            available_exchanges,
        )

    if not per_exchange:
        raise ValueError(f"Scenario {scenario.label} had no exchanges after filtering.")

    from tools.iterative_backtester import combine_analyses

    combined_metrics = combine_analyses(per_exchange)
    combined_metrics = {
        "stats": combined_metrics.get("stats", {}),
    }
    elapsed = (utc_ms() - start_ts) / 1000.0
    return ScenarioResult(
        scenario=scenario,
        per_exchange=per_exchange,
        metrics=combined_metrics,
        elapsed_seconds=elapsed,
        output_path=scenario_dir,
    )


def _run_combined_dataset(
    dataset: ExchangeDataset,
    scenario: SuiteScenario,
    scenario_config: Dict[str, Any],
    scenario_coins: List[str],
    scenario_dir: Optional[Path],
    build_payload_fn,
    execute_backtest_fn,
    post_process_fn,
) -> Dict[str, Dict[str, Any]]:
    per_exchange: Dict[str, Dict[str, Any]] = {}

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
        raise ValueError(f"Scenario {scenario.label} has no coins after applying exchange filters.")
    scenario_config["backtest"]["coins"][dataset.exchange] = list(selected_coins)
    scenario_config["backtest"]["cache_dir"][dataset.exchange] = dataset.cache_dir

    (
        hlcvs_slice,
        btc_prices,
        timestamps,
        mss_slice,
    ) = _prepare_dataset_subset(
        dataset,
        scenario_config,
        selected_coins,
        scenario.label,
    )

    payload = build_payload_fn(
        hlcvs_slice,
        mss_slice,
        scenario_config,
        dataset.exchange,
        btc_prices,
        timestamps,
    )
    fills, equities_array, analysis = execute_backtest_fn(payload, scenario_config)
    per_exchange[dataset.exchange] = analysis
    if scenario_dir is not None:
        output_dir = scenario_dir / dataset.exchange
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            post_process_fn(
                scenario_config,
                hlcvs_slice,
                fills,
                equities_array,
                btc_prices,
                analysis,
                str(output_dir),
                dataset.exchange,
                label=scenario.label,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error(
                "Scenario %s exchange %s: post-process failed (%s)",
                scenario.label,
                dataset.exchange,
                exc,
            )
    del fills
    del equities_array
    return per_exchange


def _run_multi_dataset(
    datasets: Dict[str, ExchangeDataset],
    scenario: SuiteScenario,
    scenario_config: Dict[str, Any],
    scenario_coins: List[str],
    scenario_dir: Optional[Path],
    build_payload_fn,
    execute_backtest_fn,
    post_process_fn,
    available_exchanges: List[str],
) -> Dict[str, Dict[str, Any]]:
    per_exchange: Dict[str, Dict[str, Any]] = {}
    allowed_exchanges = set(scenario.exchanges or available_exchanges)
    for exchange_key, dataset in datasets.items():
        if allowed_exchanges and dataset.exchange not in allowed_exchanges:
            continue
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

        (
            hlcvs_slice,
            btc_prices,
            timestamps,
            mss_slice,
        ) = _prepare_dataset_subset(
            dataset,
            scenario_config,
            exchange_coins,
            scenario.label,
        )

        payload = build_payload_fn(
            hlcvs_slice,
            mss_slice,
            scenario_config,
            dataset.exchange,
            btc_prices,
            timestamps,
        )
        fills, equities_array, analysis = execute_backtest_fn(payload, scenario_config)

        per_exchange[exchange_key] = analysis
        if scenario_dir is not None:
            try:
                exchange_dir = scenario_dir / dataset.exchange
                exchange_dir.mkdir(parents=True, exist_ok=True)
                post_process_fn(
                    scenario_config,
                    hlcvs_slice,
                    fills,
                    equities_array,
                    btc_prices,
                    analysis,
                    str(exchange_dir),
                    dataset.exchange,
                    label=f"{scenario.label}/{dataset.exchange}",
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

    return per_exchange


def _normalize_date_to_ts(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(f"Invalid date value: {value!r}")
    trimmed = value.strip()
    if trimmed in {"now", "today", ""}:
        trimmed = format_end_date(trimmed)
    return int(date_to_ts(trimmed))


def _compute_slice_indices(
    dataset: ExchangeDataset,
    scenario_config: Dict[str, Any],
    selected_coins: Sequence[str],
    scenario_label: str,
) -> Tuple[int, int, List[int]]:
    """
    Compute slice indices for lazy slicing from master dataset.
    Returns (start_idx, end_idx, coin_indices) without creating actual array slices.
    """
    start_value = require_config_value(scenario_config, "backtest.start_date")
    end_value = require_config_value(scenario_config, "backtest.end_date")
    start_ts = _normalize_date_to_ts(str(start_value))
    end_ts = _normalize_date_to_ts(str(end_value))
    if end_ts <= start_ts:
        raise ValueError(
            f"Scenario {scenario_label} end_date must be after start_date (got {start_value} -> {end_value})"
        )

    warmup_minutes = max(0, int(compute_backtest_warmup_minutes(scenario_config)))
    warmup_ms = warmup_minutes * 60_000
    slice_start_ts = start_ts - warmup_ms
    timestamps_arr = None
    if dataset.timestamps is not None and len(dataset.timestamps) > 0:
        timestamps_arr = np.asarray(dataset.timestamps, dtype=np.int64)
    total_steps = dataset.hlcvs.shape[0]

    if timestamps_arr is None:
        start_idx = 0
        end_idx = total_steps
    else:
        start_idx = int(np.searchsorted(timestamps_arr, slice_start_ts, side="left"))
        end_idx = int(np.searchsorted(timestamps_arr, end_ts, side="right"))
        start_idx = max(0, min(start_idx, total_steps))
        end_idx = max(start_idx + 1, min(end_idx, total_steps))
    if start_idx >= total_steps or end_idx <= start_idx:
        raise ValueError(
            f"Scenario {scenario_label} timeframe [{start_value}, {end_value}] is outside available data"
        )

    coin_indices = [dataset.coin_index[coin] for coin in selected_coins]
    return start_idx, end_idx, coin_indices


def _prepare_dataset_subset(
    dataset: ExchangeDataset,
    scenario_config: Dict[str, Any],
    selected_coins: Sequence[str],
    scenario_label: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    start_value = require_config_value(scenario_config, "backtest.start_date")
    end_value = require_config_value(scenario_config, "backtest.end_date")
    start_ts = _normalize_date_to_ts(str(start_value))
    end_ts = _normalize_date_to_ts(str(end_value))
    if end_ts <= start_ts:
        raise ValueError(
            f"Scenario {scenario_label} end_date must be after start_date (got {start_value} -> {end_value})"
        )

    warmup_minutes = max(0, int(compute_backtest_warmup_minutes(scenario_config)))
    warmup_ms = warmup_minutes * 60_000
    slice_start_ts = start_ts - warmup_ms
    timestamps_arr = None
    if dataset.timestamps is not None and len(dataset.timestamps) > 0:
        timestamps_arr = np.asarray(dataset.timestamps, dtype=np.int64)
    total_steps = dataset.hlcvs.shape[0]

    if timestamps_arr is None:
        start_idx = 0
        end_idx = total_steps
    else:
        start_idx = int(np.searchsorted(timestamps_arr, slice_start_ts, side="left"))
        end_idx = int(np.searchsorted(timestamps_arr, end_ts, side="right"))
        start_idx = max(0, min(start_idx, total_steps))
        end_idx = max(start_idx + 1, min(end_idx, total_steps))
    if start_idx >= total_steps or end_idx <= start_idx:
        raise ValueError(
            f"Scenario {scenario_label} timeframe [{start_value}, {end_value}] is outside available data"
        )

    hlcvs_window = dataset.hlcvs[start_idx:end_idx]
    btc_window = dataset.btc_usd_prices[start_idx:end_idx]
    ts_window = (
        None
        if timestamps_arr is None
        else np.asarray(timestamps_arr[start_idx:end_idx], dtype=np.int64)
    )

    indices = [dataset.coin_index[coin] for coin in selected_coins]
    hlcvs_slice = hlcvs_window[:, indices, :]

    mss_slice: Dict[str, Any] = {coin: deepcopy(dataset.mss.get(coin, {})) for coin in selected_coins}
    meta = deepcopy(dataset.mss.get("__meta__", {}))
    minute_ms = 60_000
    meta["requested_start_ts"] = int(start_ts)
    meta["requested_start_date"] = ts_to_date(int(start_ts))
    meta["requested_end_ts"] = int(end_ts)
    meta["requested_end_date"] = ts_to_date(int(end_ts))
    if ts_window is not None and len(ts_window):
        meta["effective_start_ts"] = int(ts_window[0])
        meta["effective_start_date"] = ts_to_date(int(ts_window[0]))
        meta["effective_end_ts"] = int(ts_window[-1])
        meta["effective_end_date"] = ts_to_date(int(ts_window[-1]))
        warmup_provided = max(0, int(max(0, start_ts - int(ts_window[0])) // minute_ms))
    else:
        warmup_provided = warmup_minutes
    meta["warmup_minutes_requested"] = warmup_minutes
    meta["warmup_minutes_provided"] = warmup_provided
    mss_slice["__meta__"] = meta

    interval = int(meta.get("data_interval_minutes", 1) or 1)
    offset_bars = int(meta.get("candle_interval_offset_bars", 0) or 0)
    adjustment_1m = start_idx * interval + offset_bars
    if adjustment_1m > 0:
        for coin in selected_coins:
            coin_meta = mss_slice.get(coin)
            if not isinstance(coin_meta, dict):
                continue
            if "first_valid_index" in coin_meta:
                coin_meta["first_valid_index"] = max(
                    0, int(coin_meta.get("first_valid_index", 0)) - adjustment_1m
                )
            if "last_valid_index" in coin_meta:
                coin_meta["last_valid_index"] = max(
                    0, int(coin_meta.get("last_valid_index", 0)) - adjustment_1m
                )
        if offset_bars > 0:
            meta["candle_interval_offset_bars"] = 0

    warmup_map = compute_per_coin_warmup_minutes(scenario_config)
    _recompute_index_metadata(mss_slice, hlcvs_slice, list(selected_coins), warmup_map)
    return hlcvs_slice, btc_window, ts_window, mss_slice


def _recompute_index_metadata(
    mss: Dict[str, Any], hlcvs: np.ndarray, coins: Sequence[str], warmup_map: Optional[Dict[str, int]]
) -> None:
    total_steps = hlcvs.shape[0]
    interval = int(mss.get("__meta__", {}).get("data_interval_minutes", 1) or 1)
    total_steps_1m = total_steps * interval
    warmup_map = warmup_map or {}
    default_warm = int(warmup_map.get("__default__", 0))
    for idx, coin in enumerate(coins):
        meta = mss.setdefault(coin, {})
        first_idx = int(meta.get("first_valid_index", 0))
        last_idx = int(meta.get("last_valid_index", total_steps_1m - 1))
        first_idx = max(0, min(first_idx, total_steps_1m))
        last_idx = max(0, min(last_idx, total_steps_1m - 1))
        if first_idx >= total_steps_1m:
            first_idx = total_steps_1m - 1
        if last_idx < first_idx:
            last_idx = first_idx
        if "first_valid_index" not in meta or "last_valid_index" not in meta:
            close_series = hlcvs[:, idx, 2]
            finite = np.isfinite(close_series)
            if finite.any():
                valid_indices = np.where(finite)[0]
                first_idx = int(valid_indices[0]) * interval
                last_idx = int(valid_indices[-1]) * interval + (interval - 1)
        meta["first_valid_index"] = first_idx
        meta["last_valid_index"] = last_idx
        warm_minutes = int(meta.get("warmup_minutes", warmup_map.get(coin, default_warm)))
        meta["warmup_minutes"] = warm_minutes
        if first_idx > last_idx:
            trade_start_idx = first_idx
        else:
            trade_start_idx = min(last_idx, first_idx + warm_minutes)
        meta["trade_start_index"] = trade_start_idx


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
        stats = result.metrics.get("stats", {})
        for metric, metric_stats in stats.items():
            value = metric_stats.get("mean")
            if value is None or not np.isfinite(value):
                continue
            metrics_values.setdefault(metric, []).append(float(value))

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


def build_suite_metrics_payload(
    results: Sequence[ScenarioResult], aggregate_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the canonical suite metrics payload (aggregate + per-scenario) used by both
    optimizer and backtester outputs.
    """

    scenario_metrics = {res.scenario.label: res.metrics for res in results}
    return merge_suite_payload(
        aggregate_summary.get("stats", {}),
        aggregate_values=aggregate_summary.get("aggregated", {}),
        scenario_metrics=scenario_metrics,
    )


def summarize_scenario_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a metrics dict containing only stats into a flat metric -> value map,
    preferring the mean when available.
    """

    if not isinstance(metrics, dict):
        return {}
    stats = metrics.get("stats")
    if not isinstance(stats, dict):
        return deepcopy(metrics)
    simplified: Dict[str, Any] = {}
    for name, payload in stats.items():
        if isinstance(payload, dict):
            value = payload.get("mean")
            if value is None:
                for key in ("value", "max", "min"):
                    if key in payload:
                        value = payload[key]
                        break
            simplified[name] = value
        else:
            simplified[name] = payload
    return simplified


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
    base_exchanges = require_config_value(config, "backtest.exchanges")

    base_start = require_config_value(config, "backtest.start_date")
    base_end = require_config_value(config, "backtest.end_date")
    base_coins = _flatten_coin_list(require_live_value(config, "approved_coins"))
    base_ignored = _flatten_coin_list(require_live_value(config, "ignored_coins"))

    scenarios, aggregate_cfg = build_scenarios(suite_cfg, base_exchanges=base_exchanges)

    # Determine which individual exchange datasets are needed for single-exchange scenarios
    needed_individual = _determine_needed_individual_exchanges(scenarios, base_exchanges)

    # Expand exchanges_list to include scenario-required exchanges that aren't in base
    exchanges_list = sorted(set(base_exchanges) | needed_individual)
    added_exchanges = needed_individual - set(base_exchanges)
    if added_exchanges:
        logging.info(
            "Expanded exchanges from %s to %s (added %s from scenario requirements)",
            base_exchanges,
            exchanges_list,
            sorted(added_exchanges),
        )

    for exchange in exchanges_list:
        await load_markets(exchange, verbose=False)
    await format_approved_ignored_coins(config, exchanges_list, verbose=False)

    suite_coin_sources = collect_suite_coin_sources(config, scenarios)

    master_coins = _collect_union((s.coins for s in scenarios), base_coins)
    if suite_coin_sources:
        master_coins = sorted(dict.fromkeys([*master_coins, *suite_coin_sources.keys()]))
    master_ignored = _collect_union((s.ignored_coins for s in scenarios), base_ignored)
    global_start, global_end = _collect_date_window(base_start, base_end, scenarios)

    base_config = deepcopy(config)
    base_config["backtest"]["start_date"] = global_start
    base_config["backtest"]["end_date"] = global_end
    base_config.setdefault("backtest", {})
    base_config["backtest"]["coin_sources"] = suite_coin_sources
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

    candle_interval = int(
        base_config.get("backtest", {}).get("candle_interval_minutes", 1) or 1
    )
    datasets = await prepare_master_datasets(
        base_config,
        exchanges_list,
        needed_individual_exchanges=needed_individual,
        candle_interval_minutes=candle_interval,
    )
    available_coins: set[str] = set()
    for dataset in datasets.values():
        available_coins.update(dataset.coins)
    if not available_coins:
        raise ValueError("No coins available after preparing master datasets.")

    if len(datasets) == 1 and "combined" in datasets:
        dataset_available_exchanges = datasets["combined"].available_exchanges
    else:
        dataset_available_exchanges = [ds.exchange for ds in datasets.values()]

    # Deduplicate scenarios that resolve to identical effective inputs.
    seen_signatures: Dict[str, str] = {}
    deduped: List[SuiteScenario] = []
    for scenario in scenarios:
        scenario_config_tmp, scenario_coins = apply_scenario(
            base_config,
            scenario,
            master_coins=master_coins,
            master_ignored=master_ignored,
            available_exchanges=dataset_available_exchanges,
            available_coins=available_coins,
            base_coin_sources=suite_coin_sources,
            quiet=True,
        )
        coin_exchange = _compute_effective_coin_exchange(
            scenario, scenario_coins, datasets, dataset_available_exchanges
        )
        signature = _build_scenario_signature(scenario_config_tmp, coin_exchange)
        if signature in seen_signatures:
            logging.info(
                "Skipping scenario %s (duplicate of %s)",
                scenario.label,
                seen_signatures[signature],
            )
            continue
        seen_signatures[signature] = scenario.label
        deduped.append(scenario)
    if len(deduped) != len(scenarios):
        logging.info("Scenario dedup: %d -> %d", len(scenarios), len(deduped))
    scenarios = deduped

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
            dataset_available_exchanges,
            available_coins,
            suite_dir,
            disable_plotting=disable_plotting,
            base_coin_sources=suite_coin_sources,
        )
        results.append(result)
        logging.info(
            "Scenario %s finished in %.2fs with %d metrics.",
            scenario.label,
            result.elapsed_seconds,
            len(result.metrics.get("stats", {})),
        )

    aggregate_summary = aggregate_metrics(results, aggregate_cfg)
    suite_metrics = build_suite_metrics_payload(results, aggregate_summary)
    # Persist a lean, canonical payload: shared schema + elapsed per scenario.
    summary_payload = {
        "suite_id": suite_timestamp,
        "meta": {
            "scenarios": [res.scenario.label for res in results],
            "timestamp": suite_timestamp,
        },
        "suite_metrics": suite_metrics,
        "per_scenario": {
            res.scenario.label: {
                "elapsed_seconds": res.elapsed_seconds,
                "output_path": str(res.output_path) if res.output_path else None,
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
        suite_metrics=suite_metrics,
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
        override_config = load_config(str(suite_config_path), verbose=False)
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
            raise ValueError(
                f"Suite config {suite_config_path} does not contain backtest.scenarios definition."
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
