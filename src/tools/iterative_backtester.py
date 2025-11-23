#!/usr/bin/env python3
"""
Interactive iterative backtesting helper.

Usage:
    python src/tools/iterative_backtester.py path/to/config.hjson

The script loads all OHLCV data up-front and then lets you rerun backtests
quickly after editing bot parameters. It prints a concise metrics table per run
and keeps track of the best configuration seen so far according to the
optimizer's scoring rules.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from prettytable import PrettyTable

# Ensure we can import modules from src/
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from backtest import prepare_hlcvs_mss, run_backtest  # noqa: E402
from config_utils import (  # noqa: E402
    format_config,
    load_config,
    parse_overrides,
    require_config_value,
    get_optional_config_value,
    normalize_limit_entries,
)
from logging_setup import configure_logging, resolve_log_level  # noqa: E402
from pure_funcs import calc_hash, denumpyize  # noqa: E402
from utils import (  # noqa: E402
    format_approved_ignored_coins,
    load_markets,
    make_get_filepath,
    ts_to_date,
    utc_ms,
)
from main import manage_rust_compilation  # noqa: E402
from metrics_schema import build_scenario_metrics, flatten_metric_stats  # noqa: E402
from limit_utils import expand_limit_checks, compute_limit_violation  # noqa: E402


# ---------------------------------------------------------------------------
# Constants mirroring optimizer scoring preferences
# ---------------------------------------------------------------------------
SHARED_METRIC_WEIGHTS = {
    "positions_held_per_day": 1.0,
    "position_held_hours_mean": 1.0,
    "position_held_hours_max": 1.0,
    "position_held_hours_median": 1.0,
    "position_unchanged_hours_max": 1.0,
    "loss_profit_ratio": 1.0,
    "loss_profit_ratio_w": 1.0,
    "volume_pct_per_day_avg": -1.0,
    "volume_pct_per_day_avg_w": -1.0,
    "peak_recovery_hours_pnl": 1.0,
}

CURRENCY_METRIC_WEIGHTS = {
    "adg": -1.0,
    "adg_per_exposure_long": -1.0,
    "adg_per_exposure_short": -1.0,
    "adg_w": -1.0,
    "adg_w_per_exposure_long": -1.0,
    "adg_w_per_exposure_short": -1.0,
    "calmar_ratio": -1.0,
    "calmar_ratio_w": -1.0,
    "drawdown_worst": 1.0,
    "drawdown_worst_mean_1pct": 1.0,
    "equity_balance_diff_neg_max": 1.0,
    "equity_balance_diff_neg_mean": 1.0,
    "equity_balance_diff_pos_max": 1.0,
    "equity_balance_diff_pos_mean": 1.0,
    "equity_choppiness": 1.0,
    "equity_choppiness_w": 1.0,
    "equity_jerkiness": 1.0,
    "equity_jerkiness_w": 1.0,
    "peak_recovery_hours_equity": 1.0,
    "expected_shortfall_1pct": 1.0,
    "exponential_fit_error": 1.0,
    "exponential_fit_error_w": 1.0,
    "gain": -1.0,
    "gain_per_exposure_long": -1.0,
    "gain_per_exposure_short": -1.0,
    "mdg": -1.0,
    "mdg_per_exposure_long": -1.0,
    "mdg_per_exposure_short": -1.0,
    "mdg_w": -1.0,
    "mdg_w_per_exposure_long": -1.0,
    "mdg_w_per_exposure_short": -1.0,
    "omega_ratio": -1.0,
    "omega_ratio_w": -1.0,
    "sharpe_ratio": -1.0,
    "sharpe_ratio_w": -1.0,
    "sortino_ratio": -1.0,
    "sortino_ratio_w": -1.0,
    "sterling_ratio": -1.0,
    "sterling_ratio_w": -1.0,
}

PENALTY_WEIGHT = 1e6


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ExchangeDataset:
    exchange: str
    coins: List[str]
    hlcvs: np.ndarray
    mss: Dict[str, Any]
    btc_usd_prices: np.ndarray
    timestamps: Optional[np.ndarray]
    cache_dir: str


@dataclass
class MetricInfo:
    value: Optional[float]
    resolved_key: Optional[str]
    weight: Optional[float]


@dataclass
class LimitInfo:
    metric: str
    stat: str
    mode: str
    metric_key: str
    value: Optional[float]
    bound: Optional[float] = None
    range: Optional[Tuple[float, float]] = None


@dataclass
class RunSummary:
    index: int
    timestamp: float
    score_vector: Tuple[float, ...]
    modifier: float
    combined_analysis: Dict[str, Any]
    analyses_per_exchange: Dict[str, Dict[str, Any]]
    scoring_metrics: Dict[str, MetricInfo]
    limit_metrics: List[LimitInfo]
    results_path: Path
    bot_hash: str
    bot_flat: Dict[str, Any] = field(default_factory=dict)
    param_deltas: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    duration_s: float = 0.0
    objective_vector: Tuple[float, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def build_scoring_weights() -> Dict[str, float]:
    weights = dict(SHARED_METRIC_WEIGHTS)
    for metric, weight in CURRENCY_METRIC_WEIGHTS.items():
        weights[f"{metric}_usd"] = weight
        weights[f"{metric}_btc"] = weight
        weights.setdefault(metric, weight)
        weights.setdefault(f"usd_{metric}", weight)
        weights.setdefault(f"btc_{metric}", weight)
    return weights


def ensure_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-4):
        return f"{value:.6g}"
    return f"{value:.6f}"


def summarize_limit(info: LimitInfo) -> Tuple[str, str, str]:
    """
    Returns (constraint, delta, status) for display purposes.
    """
    if info.mode == "greater_than":
        constraint = f"≤ {format_number(info.bound)}"
        if info.value is None or info.bound is None:
            return constraint, "-", "-"
        diff = info.value - info.bound
        status = "VIOL" if diff > 0 else "OK"
        return constraint, format_number(diff), status
    if info.mode == "less_than":
        constraint = f"≥ {format_number(info.bound)}"
        if info.value is None or info.bound is None:
            return constraint, "-", "-"
        diff = info.bound - info.value
        status = "VIOL" if diff > 0 else "OK"
        return constraint, format_number(diff), status
    if info.mode == "outside_range":
        low, high = info.range or (None, None)
        constraint = f"[{format_number(low)}, {format_number(high)}]"
        if info.value is None or low is None or high is None:
            return constraint, "-", "-"
        if info.value < low:
            diff = low - info.value
            return constraint, format_number(diff), "VIOL"
        if info.value > high:
            diff = info.value - high
            return constraint, format_number(diff), "VIOL"
        diff = min(info.value - low, high - info.value)
        return constraint, format_number(diff), "OK"
    if info.mode == "inside_range":
        low, high = info.range or (None, None)
        constraint = f"outside [{format_number(low)}, {format_number(high)}]"
        if info.value is None or low is None or high is None:
            return constraint, "-", "-"
        if low <= info.value <= high:
            diff = min(info.value - low, high - info.value)
            return constraint, format_number(diff), "VIOL"
        # outside the forbidden band
        if info.value < low:
            diff = low - info.value
        else:
            diff = info.value - high
        return constraint, format_number(diff), "OK"
    return "-", "-", "-"


def format_diff(current: Optional[float], reference: Optional[float]) -> str:
    if current is None or reference is None:
        return ""
    diff = current - reference
    if abs(diff) < 1e-9:
        return ""
    sign = "+" if diff >= 0 else ""
    return f"{sign}{format_number(diff)}"


def flatten_bot_config(data: Any, prefix: Tuple[str, ...] = ()) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key in sorted(data.keys()):
            flat.update(flatten_bot_config(data[key], prefix + (str(key),)))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            flat.update(flatten_bot_config(value, prefix + (str(idx),)))
    else:
        flat[".".join(prefix)] = data
    return flat


def format_param_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return format_number(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def combine_analyses(analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Build structured stats for a scenario."""

    return build_scenario_metrics(analyses)


def build_limit_checks(
    limits: Any,
    scoring_weights: Dict[str, float],
    scoring_index_map: Optional[Dict[str, List[int]]] = None,
) -> List[Dict[str, Any]]:
    normalized_limits = normalize_limit_entries(limits)
    return expand_limit_checks(
        normalized_limits,
        scoring_weights,
        penalty_weight=PENALTY_WEIGHT,
        objective_index_map=scoring_index_map,
    )


def resolve_metric_value(
    metric_name: str, combined: Dict[str, Any]
) -> Tuple[Optional[float], Optional[str]]:
    from itertools import permutations

    parts = metric_name.split("_")
    candidates: List[str] = []
    if len(parts) <= 1:
        candidates.append(metric_name)
    else:
        base, rest = parts[0], parts[1:]
        base_candidate = "_".join([base, *rest])
        candidates.append(base_candidate)
        for perm in permutations(rest):
            candidate = "_".join([base, *perm])
            candidates.append(candidate)

    extended: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            extended.append(candidate)
            seen.add(candidate)
        for suffix in ("usd", "btc"):
            with_suffix = f"{candidate}_{suffix}"
            if with_suffix not in seen:
                extended.append(with_suffix)
                seen.add(with_suffix)
            parts_candidate = candidate.split("_")
            if len(parts_candidate) >= 2:
                inserted = "_".join(parts_candidate[:-1] + [suffix, parts_candidate[-1]])
                if inserted not in seen:
                    extended.append(inserted)
                    seen.add(inserted)

    for candidate in extended:
        key = f"{candidate}_mean"
        if key in combined:
            return ensure_float(combined[key]), candidate
    return None, None


def calc_score_vector(
    scoring_keys: Iterable[str],
    combined: Dict[str, Any],
    scoring_weights: Dict[str, float],
    limit_checks: List[Dict[str, Any]],
) -> Tuple[Tuple[float, ...], float]:
    scoring_keys = list(scoring_keys)
    per_objective_modifier = [0.0] * len(scoring_keys)
    modifier = 0.0
    for check in limit_checks:
        val = ensure_float(combined.get(check["metric_key"]))
        penalty = compute_limit_violation(check, val)
        if not penalty:
            continue
        targets = check.get("objective_indexes") or []
        if targets:
            for idx in targets:
                if 0 <= idx < len(per_objective_modifier):
                    per_objective_modifier[idx] += penalty
        else:
            modifier += penalty

    scores: List[float] = []
    for idx, key in enumerate(scoring_keys):
        penalty_total = modifier + per_objective_modifier[idx]
        if penalty_total:
            scores.append(penalty_total)
            continue
        value, resolved = resolve_metric_value(key, combined)
        if value is None:
            scores.append(float("inf"))
            continue
        weight = scoring_weights.get(resolved or key)
        if weight is None:
            scores.append(value)
        elif weight < 0:
            scores.append(-value)
        else:
            scores.append(value)
    return tuple(scores), modifier


def make_backtest_signature(config: Dict[str, Any]) -> str:
    backtest = deepcopy(config.get("backtest", {}))
    backtest.pop("coins", None)
    backtest.pop("cache_dir", None)
    return calc_hash(backtest)


def format_timestamp(ms: float) -> str:
    return ts_to_date(ms)[:19].replace("T", " ")


# ---------------------------------------------------------------------------
# Core session class
# ---------------------------------------------------------------------------
class IterativeBacktestSession:
    def __init__(self, config_path: Path, log_level: Optional[str], auto_run: bool) -> None:
        self.config_path = config_path
        self.log_level = log_level
        self.auto_run = auto_run
        self.datasets: Dict[str, ExchangeDataset] = {}
        self.backtest_exchanges: List[str] = []
        self.combine_ohlcvs = False
        self.backtest_signature: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.history: List[RunSummary] = []
        self.best_run_index: Optional[int] = None
        self.config_cache: Dict[str, RunSummary] = {}
        self.last_bot_flat: Optional[Dict[str, Any]] = None
        self.scoring_keys: List[str] = []
        self.backtest_durations: List[float] = []
        self.pareto_front_indices: List[int] = []
        self.scoring_weights = build_scoring_weights()

    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        manage_rust_compilation()
        config = await self._load_config()
        self.backtest_exchanges = list(require_config_value(config, "backtest.exchanges"))
        self.combine_ohlcvs = bool(require_config_value(config, "backtest.combine_ohlcvs"))
        self.backtest_signature = make_backtest_signature(config)
        base_dir = require_config_value(config, "backtest.base_dir")
        session_label = time.strftime("iterative_%Y%m%d_%H%M%S")
        session_path = Path(make_get_filepath(os.path.join(base_dir, "iterative", session_label, "")))
        self.session_dir = Path(session_path)
        self.datasets = await self._prepare_datasets(config)
        logging.info("Loaded OHLCV data for %s", ", ".join(sorted(self.datasets.keys())))

    # ------------------------------------------------------------------
    async def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        config = load_config(str(self.config_path), verbose=False)
        config = format_config(config, verbose=False)
        config = parse_overrides(config, verbose=False)
        # Configure logging lazily based on CLI/debug preference
        level = resolve_log_level(
            self.log_level, get_optional_config_value(config, "logging.level", None), fallback=1
        )
        configure_logging(debug=level)
        config.setdefault("logging", {})
        config["logging"]["level"] = level
        # Ensure exchanges have markets loaded and live coin lists expanded
        for ex in require_config_value(config, "backtest.exchanges"):
            await load_markets(ex, verbose=False)
        await format_approved_ignored_coins(
            config, require_config_value(config, "backtest.exchanges"), verbose=False
        )
        config.setdefault("backtest", {})
        config["backtest"].setdefault("cache_dir", {})
        config["backtest"].setdefault("coins", {})
        config["disable_plotting"] = True
        return config

    # ------------------------------------------------------------------
    async def _prepare_datasets(self, config: Dict[str, Any]) -> Dict[str, ExchangeDataset]:
        datasets: Dict[str, ExchangeDataset] = {}
        if self.combine_ohlcvs:
            exchange = "combined"
            (
                coins,
                hlcvs,
                mss,
                _results_path,
                cache_dir,
                btc_usd_prices,
                timestamps,
            ) = await prepare_hlcvs_mss(config, exchange)
            config["backtest"]["coins"][exchange] = coins
            config["backtest"]["cache_dir"][exchange] = str(cache_dir)
            datasets[exchange] = ExchangeDataset(
                exchange=exchange,
                coins=coins,
                hlcvs=hlcvs,
                mss=mss,
                btc_usd_prices=btc_usd_prices,
                timestamps=timestamps,
                cache_dir=str(cache_dir),
            )
        else:
            for exchange in self.backtest_exchanges:
                exchange_config = deepcopy(config)
                (
                    coins,
                    hlcvs,
                    mss,
                    _results_path,
                    cache_dir,
                    btc_usd_prices,
                    timestamps,
                ) = await prepare_hlcvs_mss(exchange_config, exchange)
                config["backtest"]["coins"][exchange] = coins
                config["backtest"]["cache_dir"][exchange] = str(cache_dir)
                datasets[exchange] = ExchangeDataset(
                    exchange=exchange,
                    coins=coins,
                    hlcvs=hlcvs,
                    mss=mss,
                    btc_usd_prices=btc_usd_prices,
                    timestamps=timestamps,
                    cache_dir=str(cache_dir),
                )
        return datasets

    # ------------------------------------------------------------------
    async def reload_datasets(self, config: Dict[str, Any]) -> None:
        logging.info("Backtest configuration changed; reloading datasets...")
        self.history.clear()
        self.best_run_index = None
        self.config_cache.clear()
        self.last_bot_flat = None
        self.pareto_front_indices = []
        self.backtest_durations.clear()
        self.scoring_keys = []
        self.datasets = await self._prepare_datasets(config)
        self.backtest_signature = make_backtest_signature(config)
        logging.info("Datasets reloaded.")

    # ------------------------------------------------------------------
    async def run_once(self) -> Tuple[RunSummary, bool]:
        config = await self._load_config()
        current_signature = make_backtest_signature(config)
        if current_signature != self.backtest_signature:
            await self.reload_datasets(config)

        bot_section = denumpyize(deepcopy(config.get("bot", {})))
        bot_hash = calc_hash(json.dumps(bot_section, sort_keys=True))
        if bot_hash in self.config_cache:
            cached = self.config_cache[bot_hash]
            self.last_bot_flat = cached.bot_flat
            logging.info(
                "Configuration unchanged; reusing cached results from run #%d.", cached.index
            )
            return cached, True

        # Inject cached metadata
        config.setdefault("backtest", {})
        config["backtest"].setdefault("coins", {})
        config["backtest"].setdefault("cache_dir", {})
        for exchange, dataset in self.datasets.items():
            config["backtest"]["coins"][exchange] = dataset.coins
            config["backtest"]["cache_dir"][exchange] = dataset.cache_dir

        estimate = self._estimate_duration()
        progress_task: Optional[asyncio.Task] = None
        if estimate is not None or self.history:
            progress_task = asyncio.create_task(self._show_progress(estimate))

        start_time = time.perf_counter()
        analyses: Dict[str, Dict[str, Any]] = {}
        try:
            for exchange, dataset in self.datasets.items():
                fills, equities_array, analysis = await asyncio.to_thread(
                    run_backtest,
                    dataset.hlcvs,
                    dataset.mss,
                    config,
                    exchange,
                    dataset.btc_usd_prices,
                    dataset.timestamps,
                )
                analyses[exchange] = analysis
                del fills
                del equities_array
        finally:
            duration = time.perf_counter() - start_time
            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task
                self._clear_progress_line()

        combined = combine_analyses(analyses)
        combined_flat = flatten_metric_stats(combined.get("stats", {}))
        scoring_keys = list(config.get("optimize", {}).get("scoring", []))
        scoring_index_map: Dict[str, List[int]] = {}
        for idx, key in enumerate(scoring_keys):
            scoring_index_map.setdefault(key, []).append(idx)
        self.scoring_keys = scoring_keys
        limits_cfg = config.get("optimize", {}).get("limits", [])
        limit_checks = build_limit_checks(
            limits_cfg,
            self.scoring_weights,
            scoring_index_map,
        )
        score_vector, modifier = calc_score_vector(
            scoring_keys, combined_flat, self.scoring_weights, limit_checks
        )

        scoring_metrics: Dict[str, MetricInfo] = {}
        for key in scoring_keys:
            value, resolved = resolve_metric_value(key, combined_flat)
            weight = self.scoring_weights.get(resolved or key)
            scoring_metrics[key] = MetricInfo(
                value=ensure_float(value),
                resolved_key=resolved,
                weight=weight,
            )

        limit_metrics: List[LimitInfo] = []
        for check in limit_checks:
            val = ensure_float(combined_flat.get(check["metric_key"]))
            limit_metrics.append(
                LimitInfo(
                    metric=check["metric"],
                    stat=check.get("stat", ""),
                    mode=check["mode"],
                    metric_key=check["metric_key"],
                    value=val,
                    bound=check.get("bound"),
                    range=check.get("range"),
                )
            )

        current_bot_flat = flatten_bot_config(bot_section)
        param_deltas: Dict[str, Tuple[Any, Any]] = {}
        if self.last_bot_flat is not None:
            prev_keys = set(self.last_bot_flat.keys())
            new_keys = set(current_bot_flat.keys())
            for key in sorted(prev_keys | new_keys):
                prev_val = self.last_bot_flat.get(key)
                new_val = current_bot_flat.get(key)
                if prev_val != new_val:
                    param_deltas[key] = (prev_val, new_val)

        objectives: List[float] = []
        for key in scoring_keys:
            info = scoring_metrics.get(key)
            if info is None or info.value is None:
                objectives.append(float("inf"))
                continue
            weight = info.weight
            if weight is None:
                weight = self.scoring_weights.get(info.resolved_key or key)
            if weight is None:
                objectives.append(info.value)
            elif weight < 0:
                objectives.append(-info.value)
            else:
                objectives.append(info.value)

        run_index = len(self.history) + 1
        run_ts = utc_ms()
        run_dir = self._write_results(
            run_index,
            run_ts,
            analyses,
            combined,
            config,
            score_vector,
            modifier,
        )

        summary = RunSummary(
            index=run_index,
            timestamp=run_ts,
            score_vector=score_vector,
            modifier=modifier,
            combined_analysis=combined,
            analyses_per_exchange=analyses,
            scoring_metrics=scoring_metrics,
            limit_metrics=limit_metrics,
            results_path=run_dir,
            bot_hash=bot_hash,
            bot_flat=current_bot_flat,
            param_deltas=param_deltas,
            duration_s=duration,
            objective_vector=tuple(objectives),
        )
        self.history.append(summary)
        self._update_best_run(summary)
        self.config_cache[bot_hash] = summary
        self.last_bot_flat = current_bot_flat
        self.backtest_durations.append(duration)
        if len(self.backtest_durations) > 100:
            self.backtest_durations.pop(0)
        self._update_pareto_front()
        self._append_history_log(summary)
        return summary, False

    # ------------------------------------------------------------------
    def _estimate_duration(self) -> Optional[float]:
        if not self.backtest_durations:
            return None
        return sum(self.backtest_durations) / len(self.backtest_durations)

    # ------------------------------------------------------------------
    async def _show_progress(self, estimate: Optional[float]) -> None:
        estimate = estimate or (self.backtest_durations[-1] if self.backtest_durations else 1.0)
        estimate = max(estimate, 1e-6)
        bar_width = 24
        start = time.perf_counter()
        try:
            while True:
                elapsed = time.perf_counter() - start
                fraction = min(0.999, elapsed / estimate)
                filled = int(fraction * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                eta = max(0.0, estimate - elapsed)
                sys.stdout.write(
                    f"\rRunning backtest: [{bar}] {elapsed:5.1f}s elapsed, ETA {eta:5.1f}s"
                )
                sys.stdout.flush()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    def _clear_progress_line(self) -> None:
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    def _update_best_run(self, run: RunSummary) -> None:
        if self.best_run_index is None:
            self.best_run_index = run.index - 1
            return
        current_best = self.history[self.best_run_index]
        if run.score_vector < current_best.score_vector:
            self.best_run_index = run.index - 1

    # ------------------------------------------------------------------
    def _update_pareto_front(self) -> None:
        if not self.history or not self.scoring_keys:
            self.pareto_front_indices = [run.index for run in self.history]
            return

        front: List[int] = []
        objective_map = {run.index: run.objective_vector for run in self.history}
        for candidate in self.history:
            vec_candidate = objective_map.get(candidate.index)
            if vec_candidate is None or len(vec_candidate) == 0:
                front.append(candidate.index)
                continue
            dominated = False
            for other in self.history:
                if other.index == candidate.index:
                    continue
                vec_other = objective_map.get(other.index)
                if vec_other is None or len(vec_other) == 0:
                    continue
                if self._dominates(vec_other, vec_candidate):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate.index)
        self.pareto_front_indices = sorted(front)

    # ------------------------------------------------------------------
    @staticmethod
    def _dominates(vector_a: Tuple[float, ...], vector_b: Tuple[float, ...]) -> bool:
        if not vector_a or not vector_b:
            return False
        if len(vector_a) != len(vector_b):
            return False
        better_or_equal = True
        strictly_better = False
        for a, b in zip(vector_a, vector_b):
            if a > b:
                better_or_equal = False
                break
            if a < b:
                strictly_better = True
        return better_or_equal and strictly_better

    # ------------------------------------------------------------------
    def _goal_symbol(self, metric: str, info: MetricInfo) -> str:
        weight = info.weight
        if weight is None:
            weight = self.scoring_weights.get(info.resolved_key or metric)
        if weight is None:
            return "?"
        return "↑" if weight < 0 else "↓"

    # ------------------------------------------------------------------
    def _append_history_log(self, run: RunSummary) -> None:
        if self.session_dir is None:
            return
        payload: Dict[str, Any] = {
            "timestamp_ms": run.timestamp,
            "run_index": run.index,
            "score_vector": run.score_vector,
            "modifier": run.modifier,
            "results_path": str(run.results_path),
            "backtest_signature": self.backtest_signature,
            "bot_hash": run.bot_hash,
            "duration_s": run.duration_s,
            "objective_vector": run.objective_vector,
            "best_run_index": (
                (self.best_run_index + 1) if self.best_run_index is not None else run.index
            ),
            "is_best": (
                self.best_run_index == len(self.history) - 1
                if self.best_run_index is not None
                else True
            ),
        }
        payload["limits"] = [
            {
                "metric": info.metric,
                "stat": info.stat,
                "mode": info.mode,
                "metric_key": info.metric_key,
                "value": info.value,
                "bound": info.bound,
                "range": list(info.range) if info.range else None,
            }
            for info in run.limit_metrics
        ]
        payload["scoring"] = {
            name: {
                "value": info.value,
                "weight": info.weight,
                "resolved_key": info.resolved_key,
            }
            for name, info in run.scoring_metrics.items()
        }
        if run.param_deltas:
            payload["param_deltas"] = {
                key: {"previous": prev, "current": curr}
                for key, (prev, curr) in run.param_deltas.items()
            }
        log_path = self.session_dir / "history.jsonl"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True))
            fh.write("\n")

    # ------------------------------------------------------------------
    def _write_results(
        self,
        run_index: int,
        run_ts: float,
        analyses: Dict[str, Dict[str, Any]],
        combined: Dict[str, Any],
        config: Dict[str, Any],
        score_vector: Tuple[float, ...],
        modifier: float,
    ) -> Path:
        if self.session_dir is None:
            raise RuntimeError("session directory not initialised")
        timestamp_str = format_timestamp(run_ts).replace(" ", "_").replace(":", "")
        run_dir = self.session_dir / f"run_{run_index:03d}_{timestamp_str}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_ms": run_ts,
            "score_vector": score_vector,
            "modifier": modifier,
            "analysis_per_exchange": denumpyize(analyses),
            "analysis_combined": denumpyize(combined),
            "config_path": str(self.config_path),
        }
        payload_path = run_dir / "analysis.json"
        with payload_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

        config_copy = denumpyize(config)
        cfg_path = run_dir / "config_used.json"
        with cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(config_copy, fh, indent=2, sort_keys=True)

        return run_dir

    # ------------------------------------------------------------------
    def print_summary(self, run: RunSummary, reused: bool = False) -> None:
        if not self.history:
            print("No runs executed yet.")
            return
        try:
            run_pos = next(i for i, item in enumerate(self.history) if item.index == run.index)
        except StopIteration:
            run_pos = len(self.history) - 1
        prev = self.history[run_pos - 1] if run_pos > 0 else None
        best = self.history[self.best_run_index] if self.best_run_index is not None else None

        print()
        print("=" * 80)
        print(f"Run #{run.index}  @ {format_timestamp(run.timestamp)}")
        print(f"  Score vector: {tuple(f'{x:.6g}' for x in run.score_vector)}")
        if run.modifier:
            print(f"  Limit penalty applied: {run.modifier:.6g}")
        if best is not None:
            best_note = "this run" if best.index == run.index else f"run #{best.index}"
            print(f"  Best (score) so far: {best_note}")
        print(f"  Results saved to: {run.results_path}")
        if self.backtest_durations:
            avg = sum(self.backtest_durations) / len(self.backtest_durations)
            print(
                f"  Backtest duration: {run.duration_s:.3f}s "
                f"(avg {avg:.3f}s over {len(self.backtest_durations)} runs)"
            )
        else:
            print(f"  Backtest duration: {run.duration_s:.3f}s")
        if reused:
            print("  Configuration unchanged (reused cached results).")

        # Configuration delta block
        if reused:
            print("\nConfig changes vs previous run: none (reused)")
        elif prev is None:
            print("\nConfig changes vs previous run: (initial run)")
        elif not run.param_deltas:
            print("\nConfig changes vs previous run: none")
        else:
            print("\nConfig changes vs previous run:")
            changed_keys = sorted(run.param_deltas.keys())
            max_rows = 12
            for key in changed_keys[:max_rows]:
                prev_val, new_val = run.param_deltas[key]
                print(f"  {key}: {format_param_value(prev_val)} -> {format_param_value(new_val)}")
            if len(changed_keys) > max_rows:
                print(f"  … and {len(changed_keys) - max_rows} more changes")

        # Pareto front status
        if self.pareto_front_indices:
            is_pareto = run.index in self.pareto_front_indices
            status = "ON PARETO FRONT" if is_pareto else "DOMINATED"
            print(f"\nPareto status: {status}")
            pareto_runs = [r for r in self.history if r.index in self.pareto_front_indices]
            pareto_line = ", ".join(
                f"#{r.index}{'*' if r.index == run.index else ''}" for r in pareto_runs
            )
            print(f"Pareto runs: {pareto_line}")
            if not is_pareto:
                dominators = [
                    r.index
                    for r in pareto_runs
                    if r.index != run.index
                    and self._dominates(r.objective_vector, run.objective_vector)
                ]
                if dominators:
                    print(f"Dominated by: {', '.join('#' + str(idx) for idx in dominators)}")
            snapshot_table = PrettyTable()
            order = self.scoring_keys or list(run.scoring_metrics.keys())
            snapshot_table.field_names = ["Run"] + order
            max_rows = min(len(pareto_runs), 6)
            for pareto_run in pareto_runs[:max_rows]:
                row = [f"#{pareto_run.index}{'*' if pareto_run.index == run.index else ''}"]
                for metric in order:
                    info = pareto_run.scoring_metrics.get(metric)
                    row.append(format_number(info.value if info else None))
                snapshot_table.add_row(row)
            print("\nPareto front snapshot:")
            print(snapshot_table)

        # Scoring metrics table
        if run.scoring_metrics:

            def _metric_value(source: Optional[RunSummary], metric_key: str) -> Optional[float]:
                if source is None:
                    return None
                info = source.scoring_metrics.get(metric_key)
                return info.value if info else None

            order = self.scoring_keys or list(run.scoring_metrics.keys())
            table = PrettyTable()
            table.field_names = ["Metric", "Value", "ΔPrev", "ΔBest", "Goal"]
            for metric in order:
                info = run.scoring_metrics.get(metric)
                if info is None:
                    continue
                prev_val = _metric_value(prev, metric)
                best_val = _metric_value(best, metric)
                delta_prev = format_diff(info.value, prev_val) or "-"
                delta_best = format_diff(info.value, best_val) or "-"
                goal = self._goal_symbol(metric, info)
                table.add_row(
                    [
                        metric,
                        format_number(info.value),
                        delta_prev,
                        delta_best,
                        goal,
                    ]
                )
            print("\nScoring metrics:")
            print(table)

        # Limit checks table
        if run.limit_metrics:
            table = PrettyTable()
            table.field_names = ["Limit", "Value", "Constraint", "Δ", "Status"]
            rows: List[Tuple[int, List[str]]] = []
            for info in run.limit_metrics:
                label = info.metric
                if info.stat:
                    label = f"{label} ({info.stat})"
                constraint, delta, status = summarize_limit(info)
                row = [
                    label,
                    format_number(info.value),
                    constraint,
                    delta or "-",
                    status or "-",
                ]
                priority = 0 if status == "VIOL" else 1
                rows.append((priority, row))
            for _priority, row in sorted(rows, key=lambda item: (item[0], item[1][0])):
                table.add_row(row)
            print("\nLimit checks:")
            print(table)

        print("=" * 80)
        print()

    # ------------------------------------------------------------------
    def print_history(self) -> None:
        if not self.history:
            print("No runs executed yet.")
            return
        table = PrettyTable()
        table.field_names = ["Run", "Timestamp", "Score"]
        for run in self.history:
            table.add_row(
                [
                    run.index,
                    format_timestamp(run.timestamp),
                    ", ".join(f"{val:.6g}" for val in run.score_vector),
                ]
            )
        print(table)

    # ------------------------------------------------------------------
    async def interactive_loop(self) -> None:
        print("Iterative backtester ready.")
        print("Commands: [Enter] run | best | history | reload | quit")
        if self.auto_run:
            run, reused = await self.run_once()
            self.print_summary(run, reused=reused)
        while True:
            cmd = (await asyncio.to_thread(input, "iterbt> ")).strip().lower()
            if cmd in ("", "run", "r"):
                try:
                    run, reused = await self.run_once()
                    self.print_summary(run, reused=reused)
                except Exception as exc:
                    logging.exception("Backtest failed: %s", exc)
            elif cmd in ("best", "b"):
                if self.best_run_index is None:
                    print("No runs yet.")
                else:
                    self.print_summary(self.history[self.best_run_index], reused=False)
            elif cmd in ("history", "h"):
                self.print_history()
            elif cmd in ("reload",):
                try:
                    config = await self._load_config()
                    await self.reload_datasets(config)
                    print("Datasets reloaded. History cleared.")
                except Exception as exc:
                    logging.exception("Failed to reload datasets: %s", exc)
            elif cmd in ("quit", "exit", "q"):
                print("Exiting.")
                return
            elif cmd in ("help", "?"):
                print("Commands: [Enter] run | best | history | reload | quit")
            else:
                print(f"Unknown command '{cmd}'. Type 'help' for options.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Interactive iterative backtester")
    parser.add_argument("config_path", type=Path, help="Path to bot config (json/hjson)")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Override logging verbosity (warning, info, debug, trace or 0-3)",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Run a backtest immediately after loading datasets",
    )
    args = parser.parse_args()

    session = IterativeBacktestSession(args.config_path, args.log_level, args.auto_run)
    await session.initialize()
    await session.interactive_loop()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
