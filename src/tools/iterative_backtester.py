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
)
from logging_setup import configure_logging  # noqa: E402
from pure_funcs import calc_hash, denumpyize  # noqa: E402
from utils import (  # noqa: E402
    format_approved_ignored_coins,
    load_markets,
    make_get_filepath,
    ts_to_date,
    utc_ms,
)
from main import manage_rust_compilation  # noqa: E402


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
    value: Optional[float]
    bound: float
    penalize_if: str
    metric_key: str


@dataclass
class RunSummary:
    index: int
    timestamp: float
    score_vector: Tuple[float, ...]
    modifier: float
    combined_analysis: Dict[str, Any]
    analyses_per_exchange: Dict[str, Dict[str, Any]]
    scoring_metrics: Dict[str, MetricInfo]
    limit_metrics: Dict[str, LimitInfo]
    results_path: Path


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


def format_diff(current: Optional[float], reference: Optional[float]) -> str:
    if current is None or reference is None:
        return ""
    diff = current - reference
    if abs(diff) < 1e-9:
        return ""
    sign = "+" if diff >= 0 else ""
    return f"{sign}{format_number(diff)}"


def combine_analyses(analyses: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    combined: Dict[str, float] = {}
    if not analyses:
        return combined
    all_keys = set()
    for analysis in analyses.values():
        all_keys.update(analysis.keys())
    for key in sorted(all_keys):
        values = [analysis.get(key) for analysis in analyses.values()]
        if (
            not values
            or any(v is None for v in values)
            or any(isinstance(v, float) and not np.isfinite(v) for v in values)  # type: ignore[arg-type]
        ):
            combined[f"{key}_mean"] = 0.0
            combined[f"{key}_min"] = 0.0
            combined[f"{key}_max"] = 0.0
            combined[f"{key}_std"] = 0.0
        else:
            arr = np.array(values, dtype=float)
            combined[f"{key}_mean"] = float(np.mean(arr))
            combined[f"{key}_min"] = float(np.min(arr))
            combined[f"{key}_max"] = float(np.max(arr))
            combined[f"{key}_std"] = float(np.std(arr))
    return combined


def build_limit_checks(
    limits: Dict[str, float], scoring_weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    checks = []
    for full_key, bound in sorted(limits.items()):
        if full_key.startswith("penalize_if_greater_than_"):
            metric = full_key[len("penalize_if_greater_than_") :]
            penalize_if = "greater"
            suffix = "max"
        elif full_key.startswith("penalize_if_lower_than_"):
            metric = full_key[len("penalize_if_lower_than_") :]
            penalize_if = "lower"
            suffix = "min"
        else:
            metric = full_key
            weight = scoring_weights.get(metric)
            if weight is None:
                continue
            penalize_if = "lower" if weight < 0 else "greater"
            suffix = "min" if penalize_if == "lower" else "max"
        checks.append(
            {
                "metric": metric,
                "metric_key": f"{metric}_{suffix}",
                "penalize_if": penalize_if,
                "bound": float(bound),
            }
        )
    return checks


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
    modifier = 0.0
    for check in limit_checks:
        val = ensure_float(combined.get(check["metric_key"]))
        if val is None:
            continue
        if check["penalize_if"] == "greater" and val > check["bound"]:
            modifier += (val - check["bound"]) * PENALTY_WEIGHT
        elif check["penalize_if"] == "lower" and val < check["bound"]:
            modifier += (check["bound"] - val) * PENALTY_WEIGHT

    scores: List[float] = []
    for key in scoring_keys:
        if modifier:
            scores.append(modifier)
            continue
        value, resolved = resolve_metric_value(key, combined)
        if resolved is not None and resolved in scoring_weights:
            weight = scoring_weights[resolved]
        else:
            weight = scoring_weights.get(key)
        if value is None or weight is None:
            scores.append(0.0)
        else:
            scores.append(value * weight)
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
    def __init__(self, config_path: Path, debug: Optional[int], auto_run: bool) -> None:
        self.config_path = config_path
        self.debug = debug
        self.auto_run = auto_run
        self.datasets: Dict[str, ExchangeDataset] = {}
        self.backtest_exchanges: List[str] = []
        self.combine_ohlcvs = False
        self.backtest_signature: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.history: List[RunSummary] = []
        self.best_run_index: Optional[int] = None
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
        config = load_config(str(self.config_path))
        config = format_config(config, verbose=False)
        config = parse_overrides(config, verbose=False)
        # Configure logging lazily based on CLI/debug preference
        level = self.debug
        if level is None:
            level = get_optional_config_value(config, "logging.level", 1)
        try:
            level = int(float(level))
        except Exception:
            level = 1
        configure_logging(debug=level)
        # Ensure exchanges have markets loaded and live coin lists expanded
        for ex in require_config_value(config, "backtest.exchanges"):
            await load_markets(ex, verbose=False)
        await format_approved_ignored_coins(
            config, require_config_value(config, "backtest.exchanges")
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
        self.datasets = await self._prepare_datasets(config)
        self.backtest_signature = make_backtest_signature(config)
        logging.info("Datasets reloaded.")

    # ------------------------------------------------------------------
    async def run_once(self) -> RunSummary:
        config = await self._load_config()
        current_signature = make_backtest_signature(config)
        if current_signature != self.backtest_signature:
            await self.reload_datasets(config)
        # Inject cached metadata
        config.setdefault("backtest", {})
        config["backtest"].setdefault("coins", {})
        config["backtest"].setdefault("cache_dir", {})
        for exchange, dataset in self.datasets.items():
            config["backtest"]["coins"][exchange] = dataset.coins
            config["backtest"]["cache_dir"][exchange] = dataset.cache_dir

        analyses: Dict[str, Dict[str, Any]] = {}
        for exchange, dataset in self.datasets.items():
            fills, equities_array, analysis = run_backtest(
                dataset.hlcvs,
                dataset.mss,
                config,
                exchange,
                dataset.btc_usd_prices,
                dataset.timestamps,
            )
            analyses[exchange] = analysis
            # Explicitly discard large outputs we don't use
            del fills
            del equities_array

        combined = combine_analyses(analyses)
        scoring_keys = list(config.get("optimize", {}).get("scoring", []))
        limits_cfg = dict(config.get("optimize", {}).get("limits", {}))
        limit_checks = build_limit_checks(limits_cfg, self.scoring_weights)
        score_vector, modifier = calc_score_vector(
            scoring_keys, combined, self.scoring_weights, limit_checks
        )

        scoring_metrics: Dict[str, MetricInfo] = {}
        for key in scoring_keys:
            value, resolved = resolve_metric_value(key, combined)
            weight = self.scoring_weights.get(resolved or key)
            scoring_metrics[key] = MetricInfo(
                value=ensure_float(value),
                resolved_key=resolved,
                weight=weight,
            )

        limit_metrics: Dict[str, LimitInfo] = {}
        for check in limit_checks:
            val = ensure_float(combined.get(check["metric_key"]))
            limit_metrics[check["metric"]] = LimitInfo(
                value=val,
                bound=check["bound"],
                penalize_if=check["penalize_if"],
                metric_key=check["metric_key"],
            )

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
        )
        self.history.append(summary)
        self._update_best_run(summary)
        self._append_history_log(summary)
        return summary

    # ------------------------------------------------------------------
    def _update_best_run(self, run: RunSummary) -> None:
        if self.best_run_index is None:
            self.best_run_index = run.index - 1
            return
        current_best = self.history[self.best_run_index]
        if run.score_vector < current_best.score_vector:
            self.best_run_index = run.index - 1

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
            "best_run_index": (
                (self.best_run_index + 1) if self.best_run_index is not None else run.index
            ),
            "is_best": (
                self.best_run_index == len(self.history) - 1
                if self.best_run_index is not None
                else True
            ),
        }
        payload["limits"] = {
            name: {
                "value": info.value,
                "bound": info.bound,
                "penalize_if": info.penalize_if,
                "metric_key": info.metric_key,
            }
            for name, info in run.limit_metrics.items()
        }
        payload["scoring"] = {
            name: {
                "value": info.value,
                "weight": info.weight,
                "resolved_key": info.resolved_key,
            }
            for name, info in run.scoring_metrics.items()
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
    def print_summary(self, run: RunSummary) -> None:
        prev = self.history[-2] if len(self.history) > 1 else None
        best = self.history[self.best_run_index] if self.best_run_index is not None else None

        print()
        print("=" * 80)
        print(
            f"Run #{run.index}  @ {format_timestamp(run.timestamp)}  "
            f"score={tuple(f'{x:.6g}' for x in run.score_vector)}"
        )
        if run.modifier:
            print(f"  Limit penalty applied: {run.modifier:.6g}")
        if best is not None:
            best_note = "current run" if best.index == run.index else f"run #{best.index}"
            print(f"Best so far: {best_note} score={tuple(f'{x:.6g}' for x in best.score_vector)}")
        print(f"Results saved to: {run.results_path}")

        if run.scoring_metrics:

            def _get_metric_value(source: Optional[RunSummary], metric_key: str) -> Optional[float]:
                if source is None:
                    return None
                info = source.scoring_metrics.get(metric_key)
                return info.value if info else None

            table = PrettyTable()
            table.field_names = ["Metric", "Value", "ΔPrev", "ΔBest", "Goal"]
            for metric, info in run.scoring_metrics.items():
                prev_val = _get_metric_value(prev, metric)
                best_val = _get_metric_value(best, metric)
                goal = "↑" if (info.weight is not None and info.weight < 0) else "↓"
                table.add_row(
                    [
                        metric,
                        format_number(info.value),
                        format_diff(info.value, prev_val),
                        format_diff(info.value, best_val),
                        goal,
                    ]
                )
            print("\nScoring metrics:")
            print(table)

        if run.limit_metrics:
            table = PrettyTable()
            table.field_names = ["Limit", "Value", "Bound", "ΔBound", "Status"]
            for metric, info in run.limit_metrics.items():
                delta = ""
                status = "OK"
                if info.value is not None:
                    if info.penalize_if == "greater":
                        diff = info.value - info.bound
                        delta = format_number(diff)
                        status = "VIOL" if diff > 0 else "OK"
                    else:
                        diff = info.bound - info.value
                        delta = format_number(diff)
                        status = "VIOL" if diff > 0 else "OK"
                table.add_row(
                    [
                        metric,
                        format_number(info.value),
                        format_number(info.bound),
                        delta,
                        status,
                    ]
                )
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
            run = await self.run_once()
            self.print_summary(run)
        while True:
            cmd = (await asyncio.to_thread(input, "iterbt> ")).strip().lower()
            if cmd in ("", "run", "r"):
                try:
                    run = await self.run_once()
                    self.print_summary(run)
                except Exception as exc:
                    logging.exception("Backtest failed: %s", exc)
            elif cmd in ("best", "b"):
                if self.best_run_index is None:
                    print("No runs yet.")
                else:
                    self.print_summary(self.history[self.best_run_index])
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
        "--debug-level",
        type=int,
        default=None,
        help="Override logging verbosity (0=warn, 1=info, 2=debug, 3=trace)",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Run a backtest immediately after loading datasets",
    )
    args = parser.parse_args()

    session = IterativeBacktestSession(args.config_path, args.debug_level, args.auto_run)
    await session.initialize()
    await session.interactive_loop()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
