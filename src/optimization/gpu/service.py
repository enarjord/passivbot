from __future__ import annotations

from config.policy import validate_minimum_cost_filters

import copy
from dataclasses import replace
import hashlib
import json
import logging
import os
import time
from optimization.progress import duration, log_tokens, publish_progress
from optimization.gpu.replay_progress import replay_scope

import numpy as np

from config.metrics import EXPOSURE_RMS_METRICS
from config.shared_bot import flatten_shared_bot_side
from optimization.gpu.parameters import (
    _single_coin_exposure_params, _position_exposure_enforcer_params,
    _total_exposure_enforcer_params, _unstuck_params, _hsl_params,
    prepare_candidate_parameters,
)
from optimization.gpu.coin_parameters import build_coin_override_parameters
from config.validate import validate_limit_order_fill_buffer_pct
from optimizer_overrides import unstuck_ema_spans_coupled
from optimization.gpu.runtime import checkpoint_runtime, gpu_device, synchronize
from optimization.gpu.autotune import proxy_batches
from optimization.gpu.metric_registry import (
    BTC_INTRADAY_RISK_METRICS,
    EXPONENTIAL_EQUITY_METRICS,
    EXPONENTIAL_BTC_EQUITY_METRICS,
    RAW_STRATEGY_EQUITY_METRICS,
    RAW_STRATEGY_DRAWDOWN_CONSUMERS,
    ENTRY_INTERVAL_METRICS,
    EQUITY_BALANCE_DIFF_METRICS,
    HARD_STOP_PROXY_METRICS,
    HARD_STOP_LIFECYCLE_METRICS,
)
from optimization.gpu.model import (
    MARKET_EXECUTION_TICKS_ABI,
    validate_market_order_slippage_pct,
    adaptive_params,
    ADAPTIVE_OVERRIDE_KEYS,
    EMA_ANCHOR_COIN_OVERRIDE_ADAPTIVE_START,
    TRAILING_MARTINGALE_COIN_OVERRIDE_ADAPTIVE_START,
    UNSTUCK_EMA_PARAM_KEYS,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    GLOBAL_RISK_PARAM_KEYS,
    HSL_COIN_OVERRIDE_PATHS,
    MPS_MULTICOIN_MAX_COINS,
    MPS_TM_MULTICOIN_CHUNK_BARS,
    MPS_TM_MULTICOIN_CHUNK_CANDIDATE_STEPS,
    MPS_TM_MULTICOIN_CHUNK_CANDIDATES,
    CUDA_TM_MULTICOIN_CHUNK_CANDIDATES,
    CUDA_FACTUAL_HSL_REPLAY_CHUNK_BARS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_ENABLED_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_THRESHOLD_COLUMN,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    UNSTUCK_PARAM_KEYS,
    build_mps_multicoin_data,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
    validate_hsl_signal_topology,
)

from config.access import get_optional_config_value

def _noop_interrupt_check():
    return None


CORE_OUTPUT_KEYS = {
    *EXPOSURE_RMS_METRICS,
    "entry_interval_native_metrics",
    "hsl_drawdown_ema_max_portfolio",
    "hsl_drawdown_ema_mean_worst_1pct_portfolio",
    "raw_strategy_day_max_dd",
    "raw_strategy_day_end_eq",
    "raw_strategy_day_min_eq",
    "btc_day_end_eq",
    "btc_day_min_eq",
    "btc_day_max_dd",
    "equity_balance_diff_neg_max",
    "equity_balance_diff_neg_mean",
    "equity_balance_diff_pos_max",
    "equity_balance_diff_pos_mean",
    "equity_balance_diff_neg_max_btc",
    "equity_balance_diff_neg_mean_btc",
    "equity_balance_diff_pos_max_btc",
    "equity_balance_diff_pos_mean_btc",
    "day_end_eq",
    "day_min_eq",
    "day_max_dd",
    "day_volume",
    "day_has_fill",
    "day_net_pnl",
    "day_last_fill_balance",
    "day_fill_count",
    "fill_count",
    "fill_count_entry",
    "fill_count_long",
    "fills_active_days_count",
    "coin_fill_counts",
    "pnl_recovery_max_ms",
    "day_min_balance",
    "max_dd",
    "held_max_ms",
    "held_sum_ms",
    "held_sum_squared_hours",
    "gap_sum_squared_hours",
    "held_count",
    "position_unchanged_max_ms",
    "gap_hist",
    "gap_max_ms",
    "first_fill_ts",
    "last_fill_ts",
    "recovery_max_ms",
    "account_recovery_max_ms",
    "last_high_ts",
    "first_eq_ts",
    "last_eq_ts",
    "first_eq_step",
    "last_eq_step",
    "liq_step",
    "profit_sum",
    "loss_sum",
    "profit_sum_long",
    "loss_sum_long",
    "profit_sum_short",
    "loss_sum_short",
    "entry_initial_balance_pct",
    "entry_initial_balance_pct_long",
    "entry_initial_balance_pct_short",
    "entry_interval_sum_steps",
    "entry_interval_count",
    "entry_interval_max_steps",
    "entry_interval_hist",
    "total_wallet_exposure_max",
    "total_wallet_exposure_mean",
}


_GPU_PROFILE_TIMING_KEYS = (
    "candidate_materialization",
    "candidate_packing",
    "upload_and_buffer_clear",
    "cold_compilation",
    "warm_library_lookup",
    "pipeline_preparation",
    "kernel_execution",
    "device_to_host",
    "metric_reduction",
    "result_materialization",
    "host_overhead",
)

_GPU_PROFILE_RUNNER_TIMING_KEYS = (
    "candidate_packing",
    "upload_and_buffer_clear",
    "cold_compilation",
    "warm_library_lookup",
    "pipeline_preparation",
    "kernel_execution",
    "metric_reduction",
)

_GPU_DISPATCH_PROGRESS_INTERVAL_SECONDS = 60.0


def _new_gpu_dispatch_progress(
    candidate_count: int, dispatch_batch_size: int, *, adaptive=False
):
    if candidate_count <= dispatch_batch_size and not adaptive:
        return None
    started = time.monotonic()
    return {
        "candidate_count": int(candidate_count),
        "adaptive": adaptive,
        "completed_chunks": 0,
        "dispatch_batch_size": int(dispatch_batch_size),
        "started": started,
        "last_log": started,
        "total_chunks": (int(candidate_count) + int(dispatch_batch_size) - 1)
        // int(dispatch_batch_size),
    }


def _update_gpu_dispatch_progress(
    progress,
    *,
    completed_candidates: int,
    strategy: str,
) -> None:
    if progress is None:
        return
    progress["completed_chunks"] += 1
    now = time.monotonic()
    elapsed = now - float(progress["started"])
    completed_candidates = min(
        int(completed_candidates), int(progress["candidate_count"])
    )
    completed_chunks = min(
        (completed_candidates + int(progress["dispatch_batch_size"]) - 1)
        // int(progress["dispatch_batch_size"]),
        int(progress["total_chunks"]),
    )
    complete = completed_candidates >= int(progress["candidate_count"])
    if elapsed < _GPU_DISPATCH_PROGRESS_INTERVAL_SECONDS or (
        not complete
        and now - float(progress["last_log"]) < _GPU_DISPATCH_PROGRESS_INTERVAL_SECONDS
    ):
        return
    rate = completed_candidates / max(elapsed, 1.0e-12)
    remaining = int(progress["candidate_count"]) - completed_candidates
    chunk_label = (
        str(progress["completed_chunks"])
        if progress["adaptive"]
        else f"{completed_chunks}/{progress['total_chunks']}"
    )
    delivered = log_tokens(f"GPU scenario-group progress | {replay_scope()} |", [
        f"strategy={strategy}", f"batches_done={chunk_label}",
        f"scenario_evals={completed_candidates}/{int(progress['candidate_count'])}",
        f"elapsed={duration(elapsed)}", f"eta_group={duration(remaining / max(rate, 1.0e-12))}",
    ])
    publish_progress()
    if delivered:
        progress["last_log"] = now


def _gpu_profile_features(proxy, runners) -> dict[str, bool]:
    runners = tuple(runners)
    return {
        "btc_analysis": bool(getattr(proxy, "btc_analysis_enabled", False)),
        "btc_intraday_risk": bool(getattr(proxy, "btc_risk_enabled", False)),
        "equity_balance_diff": bool(
            getattr(proxy, "equity_balance_diff_enabled", False)
        ),
        "entry_interval": bool(getattr(proxy, "entry_interval_enabled", False)),
        "strategy_eq_recovery_distribution": any(
            bool(getattr(runner, "recovery_distribution_enabled", False))
            for runner in runners
        ),
        "hsl_ema_tail": any(
            bool(getattr(runner, "hsl_ema_tail_enabled", False)) for runner in runners
        ),
        "hsl_raw_drawdown": any(
            bool(getattr(runner, "hsl_raw_drawdown_enabled", False))
            for runner in runners
        ),
        "hsl_raw_tail": any(
            bool(getattr(runner, "hsl_raw_tail_enabled", False)) for runner in runners
        ),
        "hsl_diagnostics": any(
            bool(getattr(runner, "hsl_diagnostics_enabled", False))
            for runner in runners
        ),
        "coin_fill_counts": any(
            bool(getattr(runner, "collect_coin_fill_counts", False))
            for runner in runners
        ),
    }


def _new_gpu_proxy_profile(
    proxy,
    candidates,
    runners,
    *,
    coin_count,
    side_count,
    candle_count=None,
    dispatch_batch_size=None,
):
    if candle_count is None:
        candle_count = max(
            (int(getattr(runner, "n", 0)) for runner in runners), default=0
        )
    candle_count = int(candle_count)
    if dispatch_batch_size is None:
        dispatch_batch_size = int(
            getattr(proxy, "dispatch_batch_size", getattr(proxy, "batch_size", 0))
        )
    dispatch_batch_size = int(dispatch_batch_size)
    return {
        "schema_version": 1,
        "scope": "proxy_evaluation",
        "strategy": str(getattr(proxy, "strategy_kind", "unknown")),
        "candidate_count": len(candidates),
        "configured_batch_size": int(getattr(proxy, "batch_size", 0)),
        "max_dispatch_candidate_bars": int(
            getattr(
                proxy,
                "max_dispatch_candidate_bars",
                MPS_MAX_DISPATCH_CANDIDATE_BARS,
            )
        ),
        "dispatch_batch_size": dispatch_batch_size,
        "dispatch_chunk_count": (
            (len(candidates) + dispatch_batch_size - 1) // dispatch_batch_size
            if dispatch_batch_size > 0
            else 0
        ),
        "actual_dispatch_batch_sizes": [],
        "dispatch_specializations": [],
        "dispatch_chunk_wall_seconds": [],
        "dispatch_count": 0,
        "cold_dispatch_count": 0,
        "warm_dispatch_count": 0,
        "candidate_bars": (
            len(candidates) * candle_count * int(coin_count) * int(side_count)
        ),
        "kernel_candidate_bars": 0,
        "terminal_candidate_count": 0,
        "terminal_without_equity_count": 0,
        "estimated_post_terminal_candidate_bars": 0,
        "_terminal_step_fractions": [],
        "candle_count": candle_count,
        "coin_count": int(coin_count),
        "side_count": int(side_count),
        "requested_metric_features": _gpu_profile_features(proxy, runners),
        "timings_seconds": {key: 0.0 for key in _GPU_PROFILE_TIMING_KEYS},
    }


def _gpu_profile_runner_seconds(timings: dict) -> float:
    return sum(float(timings.get(key, 0.0)) for key in _GPU_PROFILE_RUNNER_TIMING_KEYS)


def _gpu_profile_unattributed_seconds(
    timings: dict,
    elapsed_seconds: float,
    *,
    device_to_host_before: float,
    runner_seconds_before: float,
) -> float:
    extra_device_to_host = float(timings["device_to_host"]) - float(
        device_to_host_before
    )
    extra_runner_seconds = _gpu_profile_runner_seconds(timings) - float(
        runner_seconds_before
    )
    return max(
        0.0,
        float(elapsed_seconds) - extra_device_to_host - extra_runner_seconds,
    )


def _add_gpu_runner_profile(
    profile: dict,
    runner,
    *,
    side_count: int = 1,
    effective_candidate_steps=None,
) -> None:
    runner_profile = getattr(runner, "last_profile", {}) or {}
    if not runner_profile:
        return
    batch_size = int(runner_profile.get("batch_size", 0))
    dispatch_count = int(runner_profile.get("dispatch_count", 1))
    cold = bool(runner_profile.get("cold", False))
    profile["actual_dispatch_batch_sizes"].extend(
        runner_profile.get("candidate_batch_sizes", [batch_size])
    )
    dispatch_specialization = runner_profile.get("dispatch_specialization")
    if dispatch_specialization is not None:
        profile["dispatch_specializations"].append(dict(dispatch_specialization))
    profile["dispatch_count"] += dispatch_count
    cold_dispatches = int(
        runner_profile.get(
            "cold_dispatch_count",
            (
                int(cold)
                if "temporal_chunk_bars" in runner_profile
                else dispatch_count if cold else 0
            ),
        )
    )
    profile["cold_dispatch_count"] += cold_dispatches
    profile["warm_dispatch_count"] += dispatch_count - cold_dispatches
    runner_steps = int(getattr(runner, "n", 0))
    if effective_candidate_steps is None:
        candidate_steps = batch_size * runner_steps
    else:
        candidate_steps_array = np.asarray(
            effective_candidate_steps, dtype=np.int64
        ).reshape(-1)
        if len(candidate_steps_array) != batch_size:
            raise RuntimeError(
                "profiled effective candidate steps do not match the dispatch batch"
            )
        candidate_steps = int(np.clip(candidate_steps_array, 0, runner_steps).sum())
    # Temporal dispatches partition one replay; they do not repeat its history.
    candidate_steps = int(
        runner_profile.get("kernel_candidate_steps", candidate_steps * dispatch_count)
    )
    profile["kernel_candidate_bars"] += (
        candidate_steps * int(getattr(runner, "n_coins", 1)) * int(side_count)
    )
    if "temporal_chunk_bars" in runner_profile:
        profile.setdefault("temporal_dispatches", []).append(
            {
                key: runner_profile[key]
                for key in (
                    "dispatch_count",
                    "temporal_chunk_bars",
                    "max_dispatch_seconds",
                    "replay_state_bytes_per_candidate",
                    "threads_per_threadgroup",
                    "adaptive_temporal_chunks",
                    "adaptive_chunk_adjustments",
                    "dispatch_duration_target_seconds",
                    "temporal_chunk_bars_min",
                    "temporal_chunk_bars_max",
                )
                if key in runner_profile
            }
        )
    timings = profile["timings_seconds"]
    timings["candidate_packing"] += float(runner_profile.get("cpu_pack_seconds", 0.0))
    timings["upload_and_buffer_clear"] += float(
        runner_profile.get("upload_and_zero_seconds", 0.0)
    ) + float(runner_profile.get("pre_dispatch_sync_seconds", 0.0))
    compile_seconds = float(runner_profile.get("compile_seconds", 0.0))
    timings["cold_compilation" if cold else "warm_library_lookup"] += compile_seconds
    timings["pipeline_preparation"] += float(runner_profile.get("pipeline_seconds", 0.0))
    timings["kernel_execution"] += float(runner_profile.get("kernel_seconds", 0.0))
    timings["metric_reduction"] += float(
        runner_profile.get("metric_decode_seconds", 0.0)
    )


def _add_gpu_terminal_profile(
    profile: dict,
    output: dict,
    *,
    interval_ms: int,
    effective_start_step: int = 0,
    effective_end_step: int,
) -> None:
    """Estimate work after irreversible single-coin candidate termination."""

    alive = output.get("alive")
    last_eq_ts = output.get("last_eq_ts")
    if alive is None or last_eq_ts is None:
        return
    alive_values = alive.detach().cpu().numpy().astype(bool, copy=False)
    terminal = ~alive_values
    terminal_count = int(terminal.sum())
    if terminal_count == 0:
        return
    last_eq_values = np.asarray(last_eq_ts.detach().cpu().numpy(), dtype=np.float64)
    terminal_last_eq = last_eq_values[terminal]
    finite = np.isfinite(terminal_last_eq)
    estimated_steps = np.ones(terminal_count, dtype=np.int64)
    if np.any(finite):
        estimated_steps[finite] = np.rint(
            terminal_last_eq[finite] / max(1, int(interval_ms))
        ).astype(np.int64) - int(effective_start_step)
    effective_step_count = max(1, int(effective_end_step) - int(effective_start_step))
    estimated_steps = np.clip(estimated_steps, 1, max(1, effective_step_count - 2))
    profile["terminal_candidate_count"] += terminal_count
    profile["terminal_without_equity_count"] += int((~finite).sum())
    profile["estimated_post_terminal_candidate_bars"] += int(
        np.maximum(effective_step_count - 2 - estimated_steps, 0).sum()
    ) * int(profile.get("side_count", 1))
    profile["_terminal_step_fractions"].extend(
        (estimated_steps / max(1, effective_step_count - 2)).tolist()
    )


def _finish_gpu_proxy_profile(profile: dict, started: float) -> dict:
    profile["dispatch_chunk_count"] = len(profile["dispatch_chunk_wall_seconds"])
    profile["actual_dispatch_batch_sizes"] = list(
        profile["actual_dispatch_batch_sizes"]
    )
    profile["wall_seconds"] = time.perf_counter() - started
    accounted = sum(profile["timings_seconds"].values())
    profile["timings_seconds"]["host_overhead"] = max(
        0.0,
        profile["wall_seconds"] - accounted,
    )
    terminal_fractions = np.asarray(
        profile.pop("_terminal_step_fractions", ()), dtype=np.float64
    )
    avoidable = int(profile["estimated_post_terminal_candidate_bars"])
    profile["estimated_post_terminal_candidate_bar_fraction"] = avoidable / max(
        1, int(profile["candidate_bars"])
    )
    profile["terminal_step_fraction_p50"] = (
        float(np.quantile(terminal_fractions, 0.50))
        if len(terminal_fractions)
        else None
    )
    profile["terminal_step_fraction_p90"] = (
        float(np.quantile(terminal_fractions, 0.90))
        if len(terminal_fractions)
        else None
    )
    return profile


DIRECTIONAL_HSL_OUTPUT_KEYS = {
    "hsl_long_enabled",
    "hsl_short_enabled",
    "hsl_triggers_long",
    "hsl_triggers_short",
    "hsl_restarts_long",
    "hsl_restarts_short",
    "hsl_tier_samples_total",
    "hsl_tier_samples_red",
    "hsl_duration_sum_steps",
    "hsl_duration_max_steps",
    "hsl_duration_count",
    "hsl_trigger_drawdown_sum",
    "hsl_trigger_drawdown_count",
    "hsl_flatten_time_sum_steps",
    "hsl_flatten_time_count",
    "hsl_restart_retrigger_count",
    "hsl_halt_to_restart_equity_loss",
    "hsl_panic_close_loss_sum",
    "hsl_panic_close_loss_max",
    "hsl_panic_loss_drawdown_min",
    "hsl_panic_loss_drawdown_sum",
    "hsl_panic_loss_drawdown_max",
    "hsl_panic_loss_drawdown_count",
    "hsl_drawdown_ema_max_long",
    "hsl_drawdown_ema_max_short",
    "hsl_strategy_eq_recovery_max_ms_long",
    "hsl_strategy_eq_recovery_max_ms_short",
    "hsl_drawdown_ema_mean_worst_1pct_long",
    "hsl_drawdown_ema_mean_worst_1pct_short",
    "hsl_drawdown_raw_max_long",
    "hsl_drawdown_raw_max_short",
    "hsl_drawdown_raw_mean_worst_1pct_long",
    "hsl_drawdown_raw_mean_worst_1pct_short",
}


def _metric_uses_btc_analysis(metric) -> bool:
    name = str(metric or "").strip().lower()
    return name.endswith("_btc") or "_btc_" in name


def _btc_daily_price_context(
    btc_prices, timestamps, *, expected_count: int, expected_days: int
) -> dict[str, np.ndarray]:
    """Reduce the canonical BTC/USD series onto the proxy's UTC-day grid."""

    btc = np.asarray(btc_prices, dtype=np.float64).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    if len(btc) != int(expected_count) or len(ts) != int(expected_count):
        raise ValueError(
            "MPS BTC analysis requires BTC prices and timestamps matching the "
            f"prepared candles: btc={len(btc)}, timestamps={len(ts)}, "
            f"candles={expected_count}"
        )
    if len(btc) == 0 or np.any(~np.isfinite(btc)) or np.any(btc <= 0.0):
        raise ValueError(
            "MPS BTC analysis requires finite positive BTC/USD prices for every candle"
        )
    day_idx = ((ts // 86_400_000) - (ts[0] // 86_400_000)).astype(np.int64)
    if np.any(day_idx < 0) or int(day_idx[-1]) + 1 != int(expected_days):
        raise ValueError(
            "MPS BTC analysis day grid disagrees with the prepared proxy timeline"
        )
    day_end = np.full(int(expected_days), np.nan, dtype=np.float64)
    for day in range(int(expected_days)):
        values = btc[day_idx == day]
        if len(values) == 0:
            raise ValueError(
                f"MPS BTC analysis is missing prepared prices for UTC day {day}"
            )
        day_end[day] = values[-1]
    return {
        "btc_day_end_price": day_end,
    }


# One MPS thread simulates one candidate across every candle stream. Long-running
# command buffers can starve WindowServer because Apple silicon shares the GPU
# with the desktop. Keep one dispatch below a configurable work envelope; callers
# may still use larger evolutionary populations and configured batches, which are
# split transparently. The default admits roughly 512 candidates per dispatch over
# two million bars, while 500 million remains the documented conservative value
# for a Mac that must stay responsive during optimization.
MPS_MAX_DISPATCH_CANDIDATE_BARS = 1_000_000_000


def _gpu_proxy_execution_checkpoint_contract(
    *,
    strategy_kind: str,
    exchange: str,
    enabled_sides,
    hlcvs,
    timestamps,
    backtest_params: dict,
    exchange_params,
    base_params,
    btc_prices=None,
) -> dict:
    """Return the prepared execution inputs which make proxy state reusable."""

    hlcv_values = np.ascontiguousarray(np.asarray(hlcvs))
    timestamp_values = np.ascontiguousarray(
        np.asarray(timestamps, dtype=np.int64).reshape(-1)
    )
    if len(timestamp_values) != int(hlcv_values.shape[0]):
        raise ValueError(
            "GPU proxy checkpoint timestamp identity disagrees with prepared "
            f"candles: timestamps={len(timestamp_values)}, candles={hlcv_values.shape[0]}"
        )
    runtime_keys = (
        "starting_balance",
        "candle_interval_minutes",
        "requested_start_timestamp_ms",
        "first_timestamp_ms",
        "first_valid_indices",
        "last_valid_indices",
        "trade_start_indices",
        "global_warmup_bars",
        "liquidation_threshold",
        "limit_order_fill_buffer_pct",
        "filter_by_min_effective_cost",
        "minimum_coin_age_days",
        "inception_timestamps_ms",
        "dynamic_wel_by_tradability",
        "hedge_mode",
        "max_realized_loss_pct",
        "unstuck_memory_ema_span_minutes",
        "loss_gate_memory_ema_span_minutes",
        "market_order_slippage_pct",
        "market_orders_allowed",
        "market_order_near_touch_threshold",
        "forager_score_hysteresis_pct",
    )
    market_keys = (
        "qty_step",
        "price_step",
        "min_qty",
        "min_cost",
        "c_mult",
        "maker_fee",
        "taker_fee",
    )
    contract = {
        "version": 8,
        "age_admission_abi": 1,
        "minimum_cost_filter_abi": 2,
        "market_execution_ticks_abi": MARKET_EXECUTION_TICKS_ABI,
        "market_ema_bootstrap_count_abi": 1,
        "hourly_ema_admission": "observed_before_first_planning",
        "strategy_kind": str(strategy_kind),
        "exchange": str(exchange),
        "coins": [str(coin) for coin in backtest_params.get("coins", [])],
        "enabled_sides": sorted(str(side) for side in enabled_sides),
        "hlcvs": {
            "shape": [int(value) for value in hlcv_values.shape],
            "dtype": str(hlcv_values.dtype.str),
            "sha256": hashlib.sha256(memoryview(hlcv_values).cast("B")).hexdigest(),
        },
        "timestamps": {
            "count": int(len(timestamp_values)),
            "first": int(timestamp_values[0]) if len(timestamp_values) else None,
            "last": int(timestamp_values[-1]) if len(timestamp_values) else None,
            "sha256": hashlib.sha256(timestamp_values.tobytes()).hexdigest(),
        },
        "backtest": {
            key: copy.deepcopy(backtest_params.get(key)) for key in runtime_keys
        },
        "markets": [
            {key: copy.deepcopy(item.get(key)) for key in market_keys}
            for item in exchange_params
        ],
        "base_params": {
            str(side): {
                str(key): float(value) for key, value in sorted((params or {}).items())
            }
            for side, params in sorted((base_params or {}).items())
        },
    }
    if backtest_params.get("equity_hard_stop_loss", {}).get("engine") == "hsl":
        contract["backtest"]["equity_hard_stop_loss"] = copy.deepcopy(
            backtest_params["equity_hard_stop_loss"]
        )
    if backtest_params.get("ew_half_life_days") is not None:
        from config.ew import EW_DEFINITION_VERSION
        contract["exponential_weighting"] = {
            "half_life_days": backtest_params["ew_half_life_days"],
            "definition_version": EW_DEFINITION_VERSION,
        }
    if btc_prices is not None:
        btc_values = np.ascontiguousarray(
            np.asarray(btc_prices, dtype=np.float64).reshape(-1)
        )
        if len(btc_values) != int(hlcv_values.shape[0]):
            raise ValueError(
                "GPU proxy checkpoint BTC identity disagrees with prepared "
                f"candles: btc={len(btc_values)}, candles={hlcv_values.shape[0]}"
            )
        contract["btc_analysis"] = {
            "count": int(len(btc_values)),
            "dtype": str(btc_values.dtype.str),
            "sha256": hashlib.sha256(btc_values.tobytes()).hexdigest(),
        }
    return contract


def _mps_dispatch_batch_size(
    requested_batch_size: int,
    *,
    n_bars: int,
    n_coins: int = 1,
    n_sides: int = 1,
    max_candidate_bars: int = MPS_MAX_DISPATCH_CANDIDATE_BARS,
) -> int:
    requested_batch_size = max(1, int(requested_batch_size))
    n_bars = max(1, int(n_bars))
    n_coins = max(1, int(n_coins))
    n_sides = max(1, int(n_sides))
    max_candidate_bars = int(max_candidate_bars)
    if max_candidate_bars <= 0:
        raise ValueError("max_candidate_bars must be greater than zero")
    per_candidate_work = n_bars * n_coins * n_sides
    if per_candidate_work > max_candidate_bars:
        raise ValueError(
            "one GPU candidate exceeds the Apple MPS dispatch safety envelope "
            f"(bars={n_bars}, coins={n_coins}, sides={n_sides}, "
            f"candidate_bars={per_candidate_work}, "
            f"max_candidate_bars={max_candidate_bars}); "
            "use a shorter date range or fewer coins"
        )
    safe_batch_size = max(1, max_candidate_bars // per_candidate_work)
    return min(requested_batch_size, safe_batch_size)


def _candle_interval_minutes(backtest_params: dict) -> int:
    raw_interval = backtest_params.get("candle_interval_minutes", 1)
    try:
        interval = float(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "GPU shared-account replay requires candle_interval_minutes to be an "
            "integer >= 1"
        ) from exc
    if not np.isfinite(interval) or interval < 1.0 or not interval.is_integer():
        raise ValueError(
            "GPU shared-account replay requires candle_interval_minutes to be an "
            "integer >= 1"
        )
    return int(interval)




def _mps_multicoin_dispatch_plan(
    strategy_kind: str,
    requested_batch_size: int,
    *,
    n_bars: int,
    n_coins: int,
    n_sides: int,
    max_candidate_bars: int,
    device: str = "mps",
    factual_hsl: bool = True,
) -> tuple[bool, int, int]:
    if factual_hsl is not True:
        raise ValueError("Retired GPU replay layout; current dispatch requires factual_hsl=True")
    if strategy_kind not in {"ema_anchor", "trailing_martingale"} or device not in {"cuda", "mps"}:
        raise ValueError("Unsupported current GPU replay strategy or device")
    temporal_chunking = True
    chunk_candidates = (
        CUDA_TM_MULTICOIN_CHUNK_CANDIDATES
        if device == "cuda"
        else MPS_TM_MULTICOIN_CHUNK_CANDIDATES
    )
    dispatch_candidates = (
        min(requested_batch_size, chunk_candidates)
        if temporal_chunking
        else requested_batch_size
    )
    dispatch_history = (
        min(
            n_bars,
            CUDA_FACTUAL_HSL_REPLAY_CHUNK_BARS,
            MPS_TM_MULTICOIN_CHUNK_CANDIDATE_STEPS // dispatch_candidates,
            max(1, max_candidate_bars // (n_coins * n_sides)),
        )
        if temporal_chunking
        else n_bars
    )
    dispatch_batch_size = _mps_dispatch_batch_size(
        dispatch_candidates,
        n_bars=dispatch_history,
        n_coins=n_coins,
        n_sides=n_sides,
        max_candidate_bars=max_candidate_bars,
    )
    return temporal_chunking, dispatch_batch_size, dispatch_history


def _log_mps_dispatch_cap(
    *,
    requested_batch_size: int,
    dispatch_batch_size: int,
    n_bars: int,
    n_coins: int,
    n_sides: int,
    max_candidate_bars: int,
) -> None:
    if dispatch_batch_size >= requested_batch_size:
        return
    logging.warning(
        "GPU dispatch safety cap active | requested_batch=%d dispatch_batch=%d "
        "bars=%d coins=%d sides=%d max_candidate_bars=%d",
        requested_batch_size,
        dispatch_batch_size,
        n_bars,
        n_coins,
        n_sides,
        max_candidate_bars,
    )


_ACCOUNT_EQUITY_RECOVERY_METRICS = {
    "peak_recovery_days_equity_usd",
    "peak_recovery_hours_equity_usd",
}

_HSL_EMA_TAIL_METRICS = {
    "drawdown_worst_mean_1pct_ema_strategy_eq",
    "drawdown_worst_mean_1pct_ema_strategy_eq_long",
    "drawdown_worst_mean_1pct_ema_strategy_eq_short",
}
_HSL_EMA_DRAWDOWN_METRICS = {
    "drawdown_worst_ema_strategy_eq",
    "drawdown_worst_ema_strategy_eq_long",
    "drawdown_worst_ema_strategy_eq_short",
}
_HSL_RAW_DRAWDOWN_METRICS = {
    "drawdown_worst_strategy_eq_long",
    "drawdown_worst_strategy_eq_short",
}
_HSL_RAW_TAIL_METRICS = {
    "drawdown_worst_mean_1pct_strategy_eq_long",
    "drawdown_worst_mean_1pct_strategy_eq_short",
}
_HSL_STRATEGY_EQ_RECOVERY_METRICS = {
    "peak_recovery_hours_strategy_eq_long",
    "peak_recovery_hours_strategy_eq_short",
    "peak_recovery_days_strategy_eq_long",
    "peak_recovery_days_strategy_eq_short",
}


def _hsl_diagnostics_needed(needed_metrics) -> bool:
    """Return whether the requested proxy surface needs HSL-only telemetry."""
    return bool(
        set(needed_metrics)
        & (
            set(HARD_STOP_PROXY_METRICS)
            | _HSL_EMA_DRAWDOWN_METRICS
            | _HSL_EMA_TAIL_METRICS
            | _HSL_RAW_DRAWDOWN_METRICS
            | _HSL_RAW_TAIL_METRICS
            | _HSL_STRATEGY_EQ_RECOVERY_METRICS
        )
    )


_STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS = {
    "strategy_eq_recovery_days_mean",
    "strategy_eq_recovery_days_median",
    "strategy_eq_recovery_days_p95",
    "strategy_eq_recovery_days_p99",
    "strategy_eq_recovery_days_mean_worst_5pct",
    "strategy_eq_recovery_days_mean_worst_1pct",
}


def mps_requested_metric_features(
    needed_metrics, *, strategy_kind: str
) -> frozenset[str]:
    """Name opt-in MPS metric paths required by a proxy metric surface."""

    metrics = set(needed_metrics)
    features = {
        "btc_analysis": any(_metric_uses_btc_analysis(metric) for metric in metrics),
        "btc_intraday_risk": bool(metrics & (BTC_INTRADAY_RISK_METRICS | EXPONENTIAL_BTC_EQUITY_METRICS)),
        "exponential_equity": bool(metrics & EXPONENTIAL_EQUITY_METRICS),
        "equity_balance_diff": bool(metrics & EQUITY_BALANCE_DIFF_METRICS),
        "entry_interval": bool(
            str(strategy_kind).strip().lower() == "trailing_martingale"
            and metrics & ENTRY_INTERVAL_METRICS
        ),
        "strategy_eq_recovery_distribution": bool(
            metrics & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS
        ),
        "raw_strategy_risk": bool(metrics & RAW_STRATEGY_DRAWDOWN_CONSUMERS),
        "raw_strategy_growth": bool(metrics & RAW_STRATEGY_EQUITY_METRICS),
        "hsl_ema_tail": bool(metrics & _HSL_EMA_TAIL_METRICS),
        "hsl_raw_drawdown": bool(
            metrics & (_HSL_RAW_DRAWDOWN_METRICS | _HSL_RAW_TAIL_METRICS)
        ),
        "hsl_raw_tail": bool(metrics & _HSL_RAW_TAIL_METRICS),
        "hsl_diagnostics": _hsl_diagnostics_needed(metrics),
        "coin_fill_counts": bool(
            metrics & {"fills_active_symbols_count", "fills_top_symbol_share"}
        ),
    }
    return frozenset(name for name, enabled in features.items() if enabled)


def _reduce_exponential_output(output, run, data, needed_metrics):
    if not set(needed_metrics) & EXPONENTIAL_EQUITY_METRICS:
        return {}, None
    from optimization.gpu.exponential_equity import exponential_equity_metrics
    return exponential_equity_metrics(output, run, data, needed_metrics, relative_timestamps=True)


def _exponential_metadata_row(metadata, index):
    def scalar(value):
        if isinstance(value, dict):
            return {key: scalar(item) for key, item in value.items()}
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            return value.item() if value.ndim == 0 else value[index].item()
        return value
    result = scalar(metadata)
    for key in ("first_return_end_ms", "last_return_end_ms"):
        if result[key] < 0:
            result[key] = None
    statuses = ("available", "insufficient_observations", "nonpositive_equity", "liquidated")
    for curve in result["curves"].values():
        curve["status"] = statuses[curve["status"]]
    return result


def _exponential_metadata_to_host(metadata):
    if isinstance(metadata, dict):
        return {key: _exponential_metadata_to_host(value) for key, value in metadata.items()}
    if hasattr(metadata, "detach"):
        return metadata.detach().cpu()
    return metadata


def _mps_strategy_eq_recovery_distribution(output: dict, needed_metrics):
    """Run the opt-in recovery postprocessor before proxy outputs leave MPS."""

    if not set(needed_metrics) & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS:
        return None
    if "strategy_eq_recovery_distribution" in output:
        from optimization.gpu.mps_kernel import MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS, torch

        if {"strategy_eq_recovery_samples", "strategy_eq_recovery_sample_interval_days"} & output.keys():
            raise RuntimeError("Ambiguous GPU recovery output")
        stats = output["strategy_eq_recovery_distribution"]
        bounds = output.get("last_eq_ts")
        if (not isinstance(stats, torch.Tensor) or not isinstance(bounds, torch.Tensor)
            or stats.device.type not in {"cuda", "mps"} or stats.device != bounds.device
            or stats.dtype != torch.float32 or stats.ndim != 2
            or bounds.shape != (stats.shape[0],)
            or stats.shape[1] != MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS):
            raise RuntimeError("Invalid compact GPU recovery output")
        return stats
    required = {
        "strategy_eq_recovery_samples",
        "strategy_eq_recovery_sample_interval_days",
    }
    if missing := required.difference(output):
        raise RuntimeError(
            "MPS strategy-equity recovery sampling output is missing: "
            + ", ".join(sorted(missing))
        )
    from optimization.gpu.mps_kernel import (
        strategy_eq_recovery_distribution_from_samples,
    )

    return strategy_eq_recovery_distribution_from_samples(
        output["strategy_eq_recovery_samples"],
        sample_interval_days=output["strategy_eq_recovery_sample_interval_days"],
    )


def _hsl_lookback_bars(backtest_params, *, hsl_enabled):
    if not hsl_enabled:
        # Inactive hsl policies consume no history. Keep 0/all inert.
        return 0
    # A constant-sized inert binding preserves the existing kernel argument
    # shape. V9 HSL owns no historical window or factual replay capacity.
    return 1




def _multicoin_exposure_eligible_coins(
    per_side_coin_overrides: dict[str, np.ndarray],
    sides,
    wallet_exposure_column: int,
) -> np.ndarray:
    coin_count = next(iter(per_side_coin_overrides.values())).shape[0]
    eligible = np.zeros(coin_count, dtype=bool)
    for side in sides:
        fixed_coin_wels = per_side_coin_overrides[side][:, wallet_exposure_column]
        eligible |= ~np.isfinite(fixed_coin_wels) | (fixed_coin_wels != 0.0)
    return eligible


def _candidate_wallet_exposure_limit_outputs(
    candidates: list[dict],
    base_limits: dict[str, float],
    *,
    torch,
) -> dict:
    """Expose each candidate's configured side TWEL to proxy metric reducers."""

    outputs = {}
    for side in ("long", "short"):
        if side not in base_limits:
            raise ValueError(f"GPU candidate TWEL context is missing {side}")
        key = f"{side}_total_wallet_exposure_limit"
        values = np.asarray(
            [float(candidate.get(key, base_limits[side])) for candidate in candidates],
            dtype=np.float64,
        )
        outputs[f"candidate_total_wallet_exposure_limit_{side}"] = torch.from_numpy(
            values
        )
    return outputs


def _candidate_position_slot_outputs(
    candidates: list[dict],
    base_n_positions: dict[str, float],
    base_limits: dict[str, float],
    *,
    torch,
) -> dict:
    """Expose Rust analysis' configured active position-slot denominators."""

    outputs = {}
    for side in ("long", "short"):
        if side not in base_n_positions or side not in base_limits:
            raise ValueError(f"GPU candidate position-slot context is missing {side}")
        n_positions_key = f"{side}_n_positions"
        limit_key = f"{side}_total_wallet_exposure_limit"
        values = []
        for candidate in candidates:
            n_positions = float(candidate.get(n_positions_key, base_n_positions[side]))
            limit = float(candidate.get(limit_key, base_limits[side]))
            values.append(n_positions if n_positions > 0.0 and limit > 0.0 else 0.0)
        outputs[f"position_slots_{side}"] = torch.from_numpy(
            np.asarray(values, dtype=np.float64)
        )
    return outputs


def _require_supported_multicoin_valid_tails(
    hlcvs, first_valid_indices, last_valid_indices
) -> None:
    """Validate declared multi-coin coverage and packed candle integrity."""

    values = np.asarray(hlcvs)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(
            "GPU multicoin proxy requires HLCVs shaped [time, coin, fields]"
        )
    candle_count = int(values.shape[0])
    starts = [int(value) for value in first_valid_indices]
    tails = [int(value) for value in last_valid_indices]
    if not starts or not tails:
        raise ValueError("GPU multicoin proxy requires at least one prepared coin")
    if len(starts) != len(tails):
        raise ValueError(
            "GPU multicoin proxy requires matching first/last valid-index counts; "
            f"first_valid_indices={starts}, last_valid_indices={tails}"
        )
    if len(starts) != int(values.shape[1]):
        raise ValueError(
            "GPU multicoin proxy valid-index counts must match the prepared "
            f"coin count; indices={len(starts)}, coins={values.shape[1]}"
        )
    windows = []
    for coin, (first_valid_idx, last_valid_idx) in enumerate(zip(starts, tails)):
        # The exact payload uses this sentinel for a prepared coin with no
        # valid candles.  It contributes neither coverage nor a forced-delist
        # surface, but it remains present in the coin-indexed tensors.
        if first_valid_idx == candle_count and last_valid_idx == candle_count - 1:
            continue
        if not 0 <= first_valid_idx <= last_valid_idx < candle_count:
            raise ValueError(
                "GPU multicoin proxy requires each first_valid_idx within its "
                "prepared valid range; "
                f"coin={coin}, first_valid_idx={first_valid_idx}, "
                f"last_valid_idx={last_valid_idx}, candle_count={candle_count}"
            )
        windows.append((first_valid_idx, last_valid_idx))
    if not windows:
        raise ValueError(
            "GPU multicoin proxy requires at least one coin with a non-empty "
            "prepared valid range"
        )
    # Validate the representation consumed by Metal. The data packer casts
    # bars to float32 and replaces non-finite packed values with zero, so a
    # finite-positive float64 value is insufficient if it overflows or
    # underflows during that conversion.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        packed_hlc = np.asarray(values[:, :, :3], dtype=np.float32)
    actual_valid = np.all(np.isfinite(packed_hlc) & (packed_hlc > 0.0), axis=2)
    for coin, last_valid_idx in enumerate(tails):
        if (
            last_valid_idx + 1400 < candle_count
            and not actual_valid[last_valid_idx, coin]
        ):
            raise ValueError(
                "GPU multicoin proxy requires each forced-delist final "
                "candle's packed float32 H/L/C values to remain finite and "
                "positive; "
                f"coin={coin}, last_valid_idx={last_valid_idx}, "
                f"packed_hlc={packed_hlc[last_valid_idx, coin].tolist()}"
            )
    for coin, first_valid_idx in enumerate(starts):
        if (
            first_valid_idx < candle_count
            and tails[coin] >= first_valid_idx
            and not actual_valid[first_valid_idx, coin]
        ):
            raise ValueError(
                "GPU multicoin proxy requires each first-valid candle's packed "
                "float32 H/L/C values to remain finite and positive; "
                f"coin={coin}, first_valid_idx={first_valid_idx}, "
                f"packed_hlc={packed_hlc[first_valid_idx, coin].tolist()}"
            )
    candle_indices = np.arange(candle_count, dtype=np.int64)[:, None]
    within_declared_window = (
        candle_indices >= np.asarray(starts, dtype=np.int64)[None, :]
    ) & (candle_indices <= np.asarray(tails, dtype=np.int64)[None, :])
    missing = np.argwhere(within_declared_window & ~actual_valid)
    if missing.size:
        candle_index, coin = (int(value) for value in missing[0])
        raise ValueError(
            "GPU multicoin proxy requires contiguous finite positive float32 H/L/C "
            "inside every declared coin window; non-finite, finite but non-positive "
            "or float32-unrepresentable values remain fail-closed; "
            f"coin index {coin}, candle_index={candle_index}, valid_windows={windows}"
        )


def _require_no_internal_invalid_hsl_candles(
    high, low, close, *, first_valid_idx: int, last_valid_idx: int
) -> None:
    first = max(0, int(first_valid_idx))
    last = min(int(last_valid_idx), len(close) - 1)
    valid = (
        np.isfinite(high[first : last + 1])
        & np.isfinite(low[first : last + 1])
        & np.isfinite(close[first : last + 1])
        & (np.asarray(close[first : last + 1]) > 0.0)
    )
    if not bool(np.all(valid)):
        first_invalid = first + int(np.flatnonzero(~valid)[0])
        raise ValueError(
            "MPS HSL currently requires contiguous valid candles "
            f"between first and last valid indices; invalid candle at {first_invalid}"
        )


def _require_no_internal_invalid_multicoin_hsl_candles(
    hlcvs, *, hsl_enabled_coins, first_valid_indices, last_valid_indices
) -> None:
    values = np.asarray(hlcvs)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(
            "MPS multi-coin HSL requires HLCVs shaped [time, coin, fields]"
        )
    coin_count = values.shape[1]
    if not (
        len(hsl_enabled_coins)
        == len(first_valid_indices)
        == len(last_valid_indices)
        == coin_count
    ):
        raise ValueError(
            "MPS multi-coin HSL valid-index counts must match the coin count"
        )
    for coin in range(coin_count):
        if not bool(hsl_enabled_coins[coin]):
            continue
        _require_no_internal_invalid_hsl_candles(
            values[:, coin, 0],
            values[:, coin, 1],
            values[:, coin, 2],
            first_valid_idx=int(first_valid_indices[coin]),
            last_valid_idx=int(last_valid_indices[coin]),
        )


def _require_exact_safe_proxy_candles(
    hlcvs,
    *,
    exposure_eligible_coins,
    first_valid_indices,
    last_valid_indices,
    require_positive_high_low: bool,
) -> None:
    """Reject missing valuation inputs in each exposure-eligible coin window."""

    values = np.asarray(hlcvs)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError("MPS proxy requires HLCVs shaped [time, coin, fields]")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        packed = np.asarray(values[:, :, :3], dtype=np.float32)
    for coin in range(values.shape[1]):
        if not bool(exposure_eligible_coins[coin]):
            continue
        first = max(0, int(first_valid_indices[coin]))
        last = min(int(last_valid_indices[coin]), values.shape[0] - 1)
        if first > last:
            continue
        packed_hlc = packed[first : last + 1, coin]
        packed_valid = np.all(np.isfinite(packed_hlc), axis=1) & (
            packed_hlc[:, 2] > 0.0
        )
        if require_positive_high_low:
            packed_valid &= (packed_hlc[:, 0] > 0.0) & (packed_hlc[:, 1] > 0.0)
        unsafe = np.flatnonzero(~packed_valid)
        if unsafe.size:
            candle_index = first + int(unsafe[0])
            raise ValueError(
                "MPS proxy requires contiguous valid internal candles; all-NaN, "
                "infinite, finite but non-positive, partially invalid, or float32-"
                "unrepresentable prices remain fail-closed; "
                f"coin index {coin}, invalid candle at {candle_index}"
            )




def _nan_min(left, right):
    """Elementwise minimum which preserves the finite operand when only one exists."""

    left_finite = left.isfinite()
    right_finite = right.isfinite()
    return left.where(
        left_finite & ~right_finite,
        right.where(right_finite & ~left_finite, left.minimum(right)),
    )


def _nan_max(left, right):
    """Elementwise maximum which preserves the finite operand when only one exists."""

    left_finite = left.isfinite()
    right_finite = right.isfinite()
    return left.where(
        left_finite & ~right_finite,
        right.where(right_finite & ~left_finite, left.maximum(right)),
    )


def _directional_entry_initial_metrics(side: str, entry_pct):
    """Expand one directional Metal value into the shared two-side surface."""

    if side not in {"long", "short"}:
        raise ValueError(f"expected long or short entry metric side, got {side!r}")
    zeros = entry_pct.new_zeros(entry_pct.shape)
    return {
        "entry_initial_balance_pct_long": entry_pct if side == "long" else zeros,
        "entry_initial_balance_pct_short": entry_pct if side == "short" else zeros,
    }


def _directional_gross_pnl_outputs(side: str, profit_sum, loss_sum):
    """Expand one directional gross-PnL pair into the shared two-side surface."""

    if side not in {"long", "short"}:
        raise ValueError(f"expected long or short gross PnL side, got {side!r}")
    other_side = "short" if side == "long" else "long"
    return {
        f"profit_sum_{side}": profit_sum,
        f"loss_sum_{side}": loss_sum,
        f"profit_sum_{other_side}": profit_sum.new_zeros(profit_sum.shape),
        f"loss_sum_{other_side}": loss_sum.new_zeros(loss_sum.shape),
    }










def _candidate_parameter_matrix(
    candidates: list[dict],
    param_keys,
    base_params: dict,
    *,
    static_overrides: dict | None = None,
    couple_unstuck_emas: bool = False,
) -> np.ndarray:
    """Pack columns directly, preserving base < candidate < fixed override precedence."""
    if not candidates:
        return np.asarray([], dtype=np.float64)
    matrix = np.empty(
        (len(candidates), len(base_params) * len(param_keys)), dtype=np.float64
    )
    for side_index, (side, base) in enumerate(base_params.items()):
        overrides = (static_overrides or {}).get(side, {})
        for column, key in enumerate(param_keys):
            if key == "unilateralness_window":
                continue
            source_key = (
                key.removeprefix("unstuck_")
                if couple_unstuck_emas
                and key in ("unstuck_ema_span_0", "unstuck_ema_span_1")
                else key
            )
            target = side_index * len(param_keys) + column
            if source_key in overrides:
                matrix[:, target] = float(overrides[source_key])
            else:
                candidate_key = (source_key if source_key in {
                    "unstuck_memory_ema_span_minutes", "loss_gate_memory_ema_span_minutes",
                } else f"{side}_{source_key}")
                matrix[:, target] = [
                    float(
                        candidate[candidate_key]
                        if candidate_key in candidate
                        else base[source_key]
                    )
                    for candidate in candidates
                ]
        if "unilateralness_window" in param_keys:
            offset = side_index * len(param_keys)
            matrix[:, offset + param_keys.index("unilateralness_window")] = np.ceil(
                20.0
                * matrix[:, offset + param_keys.index("unilateralness_ema_span_1m")]
            )
    return matrix




# Compatibility name retained for downstream imports from the EMA foundation PR.




def _prepared_multicoin_data(
    values,
    timestamps,
    *,
    runs,
    markets,
    checkpoint_contract,
    cache=None,
    limit_order_fill_buffer_pct=0.0,
    market_order_slippage_pct=0.0,
):
    """Share immutable suite tensors only for identical validated packing inputs."""
    limit_order_fill_buffer_pct = validate_limit_order_fill_buffer_pct(
        limit_order_fill_buffer_pct
    )
    market_order_slippage_pct = validate_market_order_slippage_pct(market_order_slippage_pct)
    candles = checkpoint_contract["hlcvs"]
    timeline = checkpoint_contract["timestamps"]
    key = (
        tuple(checkpoint_contract["coins"]),
        tuple(candles["shape"]),
        candles["dtype"],
        candles["sha256"],
        timeline["count"],
        timeline["first"],
        timeline["last"],
        timeline["sha256"],
        tuple(runs),
        tuple(markets),
        limit_order_fill_buffer_pct,
        MARKET_EXECUTION_TICKS_ABI,
        market_order_slippage_pct,
    )
    if cache is not None and key in cache:
        data = cache[key]
        logging.info(
            "GPU suite reusing prepared MPS market tensors | coins=%d bars=%d saved=%.2f GiB",
            data["n_coins"],
            data["n"],
            data["invariant_bytes"] / 2**30,
        )
        return data
    from optimization.gpu.residency import current_gpu_residency
    residency = current_gpu_residency()
    def build(spill_dir=None):
        kwargs = {"spill_dir": spill_dir} if spill_dir is not None else {}
        return build_mps_multicoin_data(
            values, timestamps, runs=runs, markets=markets, include_hourly_ranges=True,
            limit_order_fill_buffer_pct=limit_order_fill_buffer_pct, **kwargs,
            market_order_slippage_pct=market_order_slippage_pct,
        )
    data = (
        residency.prepare(key, build)
        if residency is not None and cache is not None
        else build()
    )
    if cache is not None:
        cache[key] = data
    return data


class SharedAccountReplay:
    """Batched multi-coin MPS proxy for the supported strategy topology."""

    def __init__(
        self,
        *,
        config: dict,
        hlcvs: np.ndarray,
        mss: dict,
        btc: np.ndarray,
        timestamps: np.ndarray,
        exchange: str,
        batch_size: int,
        needed_metrics,
        interrupt_check=None,
        max_dispatch_candidate_bars: int = MPS_MAX_DISPATCH_CANDIDATE_BARS,
        prepared_data_cache: dict | None = None,
        factual_hsl: bool = True,
        factual_capacity_hints: dict | None = None,
        automatic_scratch_budget: bool = False,
        selection=None,
    ):
        if factual_hsl is not True:
            raise ValueError("Retired GPU replay layout; current shared-account replay requires factual_hsl=True")
        from config.gpu import validate_gpu_backtest_config
        validate_gpu_backtest_config(config)
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "GPU optimization requires the optional GPU dependencies; "
                "install Passivbot with `pip install -e '.[full,gpu-mps]'` (Apple) "
                "or `pip install -e '.[full,gpu-cuda]'` (NVIDIA)"
            ) from exc
        gpu_device(torch)

        from optimization.gpu.metrics import (
            BTC_INTRADAY_RISK_METRICS,
            ENTRY_INTERVAL_METRICS,
            EQUITY_BALANCE_DIFF_METRICS,
            compute_objectives,
            validate_gpu_metric_names,
        )

        self.needed_metrics = set(validate_gpu_metric_names(
            needed_metrics, native_replay=factual_hsl, device=gpu_device(torch),
        ))
        from config.ew import validate_ew_half_life_days
        half_life = validate_ew_half_life_days(config["backtest"].get("ew_half_life_days"))
        if self.needed_metrics & EXPONENTIAL_EQUITY_METRICS and half_life is None:
            raise ValueError("Exponential (_ew) metrics require backtest.ew_half_life_days")

        from backtest import build_backtest_payload
        from optimization.gpu.mps_kernel import (
            MpsEmaAnchorMulticoinFusedRunner,
            MpsEmaAnchorMulticoinRunner,
            MpsTrailingMartingaleMulticoinFusedRunner,
            MpsTrailingMartingaleMulticoinRunner,
        )

        self._torch = torch
        self._compute_objectives = compute_objectives
        self.btc_analysis_enabled = any(
            _metric_uses_btc_analysis(metric) for metric in self.needed_metrics
        )
        self.btc_risk_enabled = bool(self.needed_metrics & (BTC_INTRADAY_RISK_METRICS | EXPONENTIAL_BTC_EQUITY_METRICS))
        self.equity_balance_diff_enabled = bool(
            self.needed_metrics & EQUITY_BALANCE_DIFF_METRICS
        )
        btc_values = np.ascontiguousarray(np.asarray(btc, dtype=np.float64).reshape(-1))
        self.batch_size = max(1, int(batch_size))
        self.max_dispatch_candidate_bars = int(max_dispatch_candidate_bars)
        self.interrupt_check = interrupt_check or _noop_interrupt_check
        self.profile_enabled = os.environ.get(
            "PASSIVBOT_GPU_PROFILE", ""
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.last_profile: dict = {}

        values = np.asarray(hlcvs)
        if values.ndim != 3:
            raise ValueError(
                "expected multicoin HLCVs with three dimensions, " f"got {values.shape}"
            )
        coin_count = int(values.shape[1])
        if not (1 <= coin_count <= MPS_MULTICOIN_MAX_COINS):
            raise ValueError(
                f"MPS multicoin proxy supports 1..{MPS_MULTICOIN_MAX_COINS} coins; "
                f"got {coin_count}"
            )
        self.strategy_kind = (
            str(get_optional_config_value(config, 'bot.portfolio.strategy_kind', "")).strip().lower()
        )
        if self.strategy_kind not in {"ema_anchor", "trailing_martingale"}:
            raise ValueError(
                "MPS multicoin proxy supports ema_anchor or "
                f"trailing_martingale, got {self.strategy_kind!r}"
            )
        self.entry_interval_enabled = bool(
            self.strategy_kind == "trailing_martingale"
            and self.needed_metrics & ENTRY_INTERVAL_METRICS
        )
        self.native_factual_hsl = bool(factual_hsl)
        self.param_keys = (
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
            if self.strategy_kind == "trailing_martingale"
            else EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        )
        enabled_sides = [
            side for side in ("long", "short") if gpu_side_enabled(config, side)
        ]
        if len(enabled_sides) not in (1, 2):
            raise ValueError("MPS multicoin proxy requires one or two enabled sides")
        self.sides = enabled_sides
        self.temporal_chunking, self.dispatch_batch_size, dispatch_history = (
            _mps_multicoin_dispatch_plan(
                self.strategy_kind,
                self.batch_size,
                n_bars=len(values),
                n_coins=coin_count,
                n_sides=len(enabled_sides),
                max_candidate_bars=self.max_dispatch_candidate_bars,
                device=gpu_device(torch),
                factual_hsl=factual_hsl,
            )
        )
        if self.temporal_chunking:
            logging.info(
                "GPU MPS temporal replay | candidate_batch=%d chunk_bars<=%d "
                "history_bars=%d coins=%d max_candidate_bars=%d",
                self.dispatch_batch_size,
                dispatch_history,
                len(values),
                coin_count,
                self.max_dispatch_candidate_bars,
            )
        else:
            _log_mps_dispatch_cap(
                requested_batch_size=self.batch_size,
                dispatch_batch_size=self.dispatch_batch_size,
                n_bars=len(values),
                n_coins=coin_count,
                n_sides=len(enabled_sides),
                max_candidate_bars=self.max_dispatch_candidate_bars,
            )
        self.shared_account_fused = len(self.sides) == 2
        self.shared_account_proxy_mode = (
            "shared-account-fused-tm-v1"
            if self.strategy_kind == "trailing_martingale"
            else "shared-account-fused-ema-v1"
        )
        payload = build_backtest_payload(
            np.ascontiguousarray(values),
            mss,
            copy.deepcopy(config),
            exchange,
            btc_values,
            timestamps,
            selection=selection,
            metrics_only=True,
            skip_btc_analysis=not self.btc_analysis_enabled,
        )
        if not (
            len(payload.bot_params_list)
            == len(payload.strategy_params_list)
            == len(payload.exchange_params)
            == coin_count
        ):
            raise ValueError(
                "MPS multicoin payload length disagrees with prepared coin count: "
                f"coins={coin_count}, bots={len(payload.bot_params_list)}, "
                f"strategies={len(payload.strategy_params_list)}, "
                f"markets={len(payload.exchange_params)}"
            )
        backtest_params = payload.backtest_params
        limit_order_fill_buffer_pct = validate_limit_order_fill_buffer_pct(
            backtest_params.get("limit_order_fill_buffer_pct", 0.0)
        )
        from optimization.gpu.hsl import project_bot

        hsl_config = backtest_params.get("equity_hard_stop_loss", {})
        self.hsl_signal_mode = hsl_config["mode"]
        projected = [
            {side: project_bot(payload, c, side, config) for side in ("long", "short")}
            for c in range(coin_count)
        ]

        candle_interval_minutes = _candle_interval_minutes(backtest_params)
        self.dynamic_wel_by_tradability = bool(
            backtest_params.get("dynamic_wel_by_tradability", True)
        )
        self.forager_score_hysteresis_pct = float(
            backtest_params.get("forager_score_hysteresis_pct", 0.0) or 0.0
        )
        if not np.isfinite(self.forager_score_hysteresis_pct) or (
            self.forager_score_hysteresis_pct < 0.0
        ):
            raise ValueError(
                "MPS multicoin proxy requires a finite non-negative "
                "bot.global.forager.score_hysteresis_margin"
            )
        _require_supported_multicoin_valid_tails(
            values,
            backtest_params["first_valid_indices"],
            backtest_params["last_valid_indices"],
        )

        comparable_bot_keys = (
            "total_wallet_exposure_limit",
            "filter_volume_ema_span_1m",
            "filter_volatility_ema_span_1m",
            "forager_score_weights",
            "n_positions",
            "risk_twel_entry_gate_enabled",
            "risk_twel_enforcer_enabled",
            "risk_twel_enforcer_policy",
            "risk_twel_enforcer_threshold",
        )
        signal_mode = self.hsl_signal_mode
        hsl_enabled_sides = [
            side
            for side in self.sides
            if any(bool(item[side].get("hsl_enabled")) for item in projected)
        ]
        if hsl_enabled_sides:
            validate_hsl_signal_topology(
                signal_mode,
                coin_count=coin_count,
                enabled_side_count=len(self.sides),
                shared_account_controller=self.shared_account_fused,
            )
            _require_no_internal_invalid_multicoin_hsl_candles(
                values,
                hsl_enabled_coins=[
                    any(
                        bool(item[side].get("hsl_enabled"))
                        for side in hsl_enabled_sides
                    )
                    for item in projected
                ],
                first_valid_indices=backtest_params["first_valid_indices"],
                last_valid_indices=backtest_params["last_valid_indices"],
            )
        self.base_total_wallet_exposure_limits = {
            side: float(payload.bot_params_list[0][side]["total_wallet_exposure_limit"])
            for side in ("long", "short")
        }
        self.base_n_positions = {
            side: float(payload.bot_params_list[0][side]["n_positions"])
            for side in ("long", "short")
        }
        self.couple_unstuck_emas = unstuck_ema_spans_coupled(config)
        base_parameters = prepare_candidate_parameters(config, mss, exchange)
        self.base_params = {
            side: {key: base_parameters[key if key in GLOBAL_RISK_PARAM_KEYS else f"{side}_{key}"]
                   for key in self.param_keys}
            for side in self.sides
        }
        for side in self.sides:
            first_bot = projected[0][side]
            for coin in range(1, coin_count):
                bot = payload.bot_params_list[coin][side]
                if any(
                    bot.get(key) != first_bot.get(key) for key in comparable_bot_keys
                ):
                    raise ValueError(
                        "MPS multicoin proxy requires identical global "
                        f"{side} forager/risk settings across coins"
                    )

        self.checkpoint_contract = _gpu_proxy_execution_checkpoint_contract(
            strategy_kind=self.strategy_kind,
            exchange=exchange,
            enabled_sides=self.sides,
            hlcvs=values,
            timestamps=timestamps,
            backtest_params=backtest_params,
            exchange_params=payload.exchange_params,
            base_params=self.base_params,
            btc_prices=btc_values if self.btc_analysis_enabled else None,
        )

        self.checkpoint_contract.update(checkpoint_runtime(torch))

        coins = list(backtest_params.get("coins") or [])
        if len(coins) != coin_count:
            raise ValueError(
                "MPS multicoin payload coin identity disagrees with prepared data: "
                f"coins={coins}, prepared={coin_count}"
            )
        per_side_coin_overrides = {}
        per_side_override_contracts = {}
        for side in self.sides:
            overrides, contract = build_coin_override_parameters(
                strategy_kind=self.strategy_kind, config=config, mss=mss,
                exchange=exchange, coins=coins, payload=payload, side=side,
            )
            per_side_coin_overrides[side] = overrides
            per_side_override_contracts[side] = contract
        if len(self.sides) == 1:
            self.coin_override_contract = per_side_override_contracts[self.sides[0]]
        else:
            self.coin_override_contract = {
                "exchange": exchange,
                "coins": coins,
                "sides": list(self.sides),
                "values_by_side": {
                    side: per_side_override_contracts[side]["values"]
                    for side in self.sides
                },
                "exact_overrides_by_side": {
                    side: per_side_override_contracts[side]["exact_overrides"]
                    for side in self.sides
                },
                "proxy_mode": (
                    self.shared_account_proxy_mode
                    if self.shared_account_fused
                    else "independent-side-hedge-v1"
                ),
            }
            if hsl_enabled_sides:
                self.coin_override_contract["hsl_proxy_mode"] = (
                    self.shared_account_proxy_mode
                    if self.shared_account_fused
                    else "independent-pside-v1"
                )
                self.coin_override_contract["hsl_signal_mode"] = (
                    str(signal_mode).strip().lower()
                )
        self.coin_override_contract["forager_score_hysteresis_pct"] = (
            self.forager_score_hysteresis_pct
        )

        markets = [
            ProxyMarket(
                qty_step=float(item["qty_step"]),
                price_step=float(item["price_step"]),
                min_qty=float(item["min_qty"]),
                min_cost=float(item["min_cost"]),
                c_mult=float(item["c_mult"]),
                maker_fee=float(item["maker_fee"]),
                taker_fee=float(item["taker_fee"]),
            )
            for item in payload.exchange_params
        ]
        interval_ms = candle_interval_minutes * 60_000
        from optimization.gpu.model import native_age_admission_indices
        age_indices = native_age_admission_indices(timestamps, backtest_params, coin_count)
        runs = [
            ProxyRun(
                starting_balance=float(backtest_params["starting_balance"]),
                warmup_bars=max(
                    1, int(backtest_params.get("global_warmup_bars", 0) or 1)
                ),
                trade_start_idx=int(backtest_params["trade_start_indices"][coin]),
                requested_start_ts_ms=int(
                    backtest_params["requested_start_timestamp_ms"]
                ),
                guard_ts_ms=int(
                    max(
                        backtest_params["requested_start_timestamp_ms"],
                        backtest_params["first_timestamp_ms"],
                    )
                ),
                first_ts_ms=int(backtest_params["first_timestamp_ms"]),
                interval_ms=interval_ms,
                liquidation_threshold=float(
                    backtest_params.get("liquidation_threshold", 0.05)
                ),
                first_valid_idx=int(backtest_params["first_valid_indices"][coin]),
                last_valid_idx=int(backtest_params["last_valid_indices"][coin]),
                age_eligible_indices=age_indices[coin],
            )
            for coin in range(coin_count)
        ]
        self.run = replace(
            runs[0],
            first_valid_idx=min(run.first_valid_idx for run in runs),
            last_valid_idx=max(run.last_valid_idx for run in runs),
            trade_start_idx=min(run.trade_start_idx for run in runs),
        )
        if factual_hsl:
            from utils import date_to_ts, format_end_date
            self.run = replace(self.run, requested_end_ts_ms=int(
                date_to_ts(format_end_date(config["backtest"]["end_date"]))
            ))
        wallet_exposure_column = (
            EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
            if self.strategy_kind == "ema_anchor"
            else len(TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS) + 1
        )
        exposure_eligible_coins = _multicoin_exposure_eligible_coins(
            per_side_coin_overrides,
            self.sides,
            wallet_exposure_column,
        )
        _require_exact_safe_proxy_candles(
            values,
            exposure_eligible_coins=exposure_eligible_coins,
            first_valid_indices=backtest_params["first_valid_indices"],
            last_valid_indices=backtest_params["last_valid_indices"],
            require_positive_high_low=True,
        )
        self.data = _prepared_multicoin_data(
            values,
            timestamps,
            runs=runs,
            markets=markets,
            checkpoint_contract=self.checkpoint_contract,
            cache=prepared_data_cache,
            limit_order_fill_buffer_pct=limit_order_fill_buffer_pct,
            market_order_slippage_pct=backtest_params.get("market_order_slippage_pct", 0.0),
        )
        from optimization.gpu.residency import current_gpu_residency
        residency = current_gpu_residency()
        self._cuda_residency = residency if residency is not None and residency.owns(self.data) else None
        self.metrics_data = {
            "ts0": self.data["ts0"],
            "n": self.data["n"],
            "strategy_kind": self.strategy_kind,
            "ew_half_life_days": config["backtest"].get("ew_half_life_days"),
        }
        if self.btc_analysis_enabled:
            self.metrics_data.update(
                _btc_daily_price_context(
                    btc_values,
                    timestamps,
                    expected_count=self.data["n"],
                    expected_days=self.data["n_days"],
                )
            )
            self.metrics_data["btc_prices"] = btc_values
        self.runners = {}
        self.fused_runner = None
        self._runner_specs = {}
        # Small service-owned estimates outlive residency eviction. This mapping
        # contains integers only, never runners, tensors or checkpoint state.
        self._factual_capacity_hints = factual_capacity_hints
        self._automatic_scratch_budget = automatic_scratch_budget
        if selection is None and config["backtest"].get("organillo_csv") is not None:
            from organillo import prepare_selection
            selection = prepare_selection(config, backtest_params["coins"], timestamps)
        common_runner_kwargs = {
            "forager_score_hysteresis_pct": self.forager_score_hysteresis_pct,
            "max_realized_loss_pct": float(
                backtest_params.get("max_realized_loss_pct", 1.0)
            ),
            "collect_coin_fill_counts": bool(
                self.needed_metrics
                & {"fills_active_symbols_count", "fills_top_symbol_share"}
            ),
            "filter_by_min_effective_cost": validate_minimum_cost_filters(
                backtest_params["filter_by_min_effective_cost"]
            ),
            "market_order_slippage_pct": float(
                backtest_params.get("market_order_slippage_pct", 0.0)
            ),
            "market_orders_allowed": bool(
                backtest_params.get("market_orders_allowed", False)
            ),
            "market_order_near_touch_threshold": float(
                backtest_params.get("market_order_near_touch_threshold", 0.001)
            ),
            "hsl_ema_tail_enabled": bool(self.needed_metrics & _HSL_EMA_TAIL_METRICS),
            "hsl_raw_drawdown_enabled": bool(
                self.needed_metrics
                & (_HSL_RAW_DRAWDOWN_METRICS | _HSL_RAW_TAIL_METRICS)
            ),
            "hsl_raw_tail_enabled": bool(self.needed_metrics & _HSL_RAW_TAIL_METRICS),
            "recovery_distribution_enabled": bool(
                self.needed_metrics & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS
            ),
            "raw_strategy_risk_enabled": bool(
                self.needed_metrics & RAW_STRATEGY_DRAWDOWN_CONSUMERS
            ),
            "raw_strategy_growth_enabled": bool(
                self.needed_metrics & RAW_STRATEGY_EQUITY_METRICS
            ),
            "dynamic_wel_by_tradability": self.dynamic_wel_by_tradability,
            "btc_prices": (
                btc_values
                if self.btc_risk_enabled or self.equity_balance_diff_enabled
                else None
            ),
            "btc_risk_enabled": self.btc_risk_enabled,
            "equity_balance_diff_enabled": self.equity_balance_diff_enabled,
            "entry_interval_enabled": self.entry_interval_enabled,
        }
        if selection is not None:
            self.checkpoint_contract["organillo"] = selection.identity
            common_runner_kwargs["selection"] = selection
        # Realized consumers have independent request-owned memory spans.
        common_runner_kwargs["interrupt_check"] = self.interrupt_check
        common_runner_kwargs["factual_hsl"] = factual_hsl
        common_runner_kwargs["exposure_rms_enabled"] = bool(
            self.needed_metrics & EXPOSURE_RMS_METRICS
        )
        common_runner_kwargs["compact_recovery_output"] = bool(
            factual_hsl and common_runner_kwargs["recovery_distribution_enabled"]
        )
        common_runner_kwargs.update(
            pnl_lookback_bars=_hsl_lookback_bars(
                backtest_params, hsl_enabled=bool(hsl_enabled_sides)
            )
        )
        if self.shared_account_fused:
            fused_runner_cls = (
                MpsTrailingMartingaleMulticoinFusedRunner
                if self.strategy_kind == "trailing_martingale"
                else MpsEmaAnchorMulticoinFusedRunner
            )
            fused_kwargs = dict(
                long_coin_overrides=per_side_coin_overrides["long"],
                short_coin_overrides=per_side_coin_overrides["short"],
                hsl_panic_market_long=str(
                    (project_bot(payload, 0, "long", config, base=True)).get(
                        "hsl_panic_close_order_type", "limit"
                    )
                )
                .strip()
                .lower()
                == "market",
                hsl_panic_market_short=str(
                    (project_bot(payload, 0, "short", config, base=True)).get(
                        "hsl_panic_close_order_type", "limit"
                    )
                )
                .strip()
                .lower()
                == "market",
                hedge_mode=bool(backtest_params["hedge_mode"]),
                **common_runner_kwargs,
            )
            if self.temporal_chunking:
                fused_kwargs["max_dispatch_candidate_bars"] = self.max_dispatch_candidate_bars
                fused_kwargs["interrupt_check"] = self.interrupt_check
            self._runner_specs["fused"] = (fused_runner_cls, fused_kwargs)
            if self._cuda_residency is None:
                self.fused_runner = self._construct_runner("fused", fused_runner_cls, fused_kwargs)
        else:
            runner_cls = (
                MpsTrailingMartingaleMulticoinRunner
                if self.strategy_kind == "trailing_martingale"
                else MpsEmaAnchorMulticoinRunner
            )
            for side in self.sides:
                runner_kwargs = {
                    "side": side,
                    "hsl_panic_market": str(
                        (project_bot(payload, 0, side, config, base=True)).get(
                            "hsl_panic_close_order_type", "limit"
                        )
                    )
                    .strip()
                    .lower()
                    == "market",
                    **common_runner_kwargs,
                }
                runner_kwargs["coin_overrides"] = per_side_coin_overrides[side]
                if self.temporal_chunking:
                    runner_kwargs["max_dispatch_candidate_bars"] = (
                        self.max_dispatch_candidate_bars
                    )
                    runner_kwargs["interrupt_check"] = self.interrupt_check
                self._runner_specs[side] = (runner_cls, runner_kwargs)
                if self._cuda_residency is None:
                    self.runners[side] = self._construct_runner(side, runner_cls, runner_kwargs)
        if self._cuda_residency is not None:
            self._cuda_residency.register(self)

    def _create_runners(self):
        for side, (runner_cls, kwargs) in self._runner_specs.items():
            runner = self._construct_runner(side, runner_cls, kwargs)
            if side == "fused":
                self.fused_runner = runner
            else:
                self.runners[side] = runner

    def _construct_runner(self, key, runner_cls, kwargs):
        runner = runner_cls(self.run, self.data, **kwargs)
        if self._automatic_scratch_budget and runner.bars.device.type == "cuda":
            from optimization.gpu.autotune import cuda_history_budget

            runner.automatic_scratch_budget = True
            free, total = map(int, self._torch.cuda.mem_get_info())
            runner.hsl_scratch_budget_bytes = cuda_history_budget(
                runner, free, total,
                owners=1 if self.shared_account_fused else len(self.sides),
            )
        return runner

    def _remember_factual_capacities(self):
        # V9 forward risk has no historical capacity to learn.
        return

    def _parameter_matrix(
        self, candidates: list[dict], side: str | None = None
    ) -> np.ndarray:
        if side is None:
            if len(self.sides) != 1:
                raise ValueError("side is required for dual-side multicoin parameters")
            side = self.sides[0]
        if self.hsl_signal_mode == "portfolio":
            candidates = [dict(c) for c in candidates]
            for candidate in candidates:
                for key in (
                    "hsl_red_threshold",
                    "hsl_drawdown_ratio_ema_span_minutes",
                    "hsl_cooldown_ms_lo", "hsl_cooldown_ms_hi",
                ):
                    if key in candidate:
                        candidate[f"{side}_{key}"] = candidate[key]
        param_keys = getattr(self, "param_keys", EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        return _candidate_parameter_matrix(
            candidates,
            param_keys,
            {side: self.base_params[side]},
            couple_unstuck_emas=getattr(self, "couple_unstuck_emas", False),
        )

    def suite_batch_key(self):
        """Identify GPU scenarios differing only in materializable parameters.

        The receiving proxy still splits the combined candidates at its device's
        dispatch cap; grouping scenarios never expands the work envelope.
        """
        if (
            gpu_device(self._torch) not in {"cuda", "mps"}
            or self.strategy_kind != "trailing_martingale"
            or len(self.sides) != 1
        ):
            return None
        runner_cls, kwargs = self._runner_specs[self.sides[0]]
        runner_settings = {}
        for key, value in kwargs.items():
            if key == "selection":
                runner_settings[key] = value.identity
            elif isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                runner_settings[key] = (
                    array.shape,
                    array.dtype.str,
                    hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
                )
            elif key == "interrupt_check":
                runner_settings[key] = id(value)
            else:
                runner_settings[key] = value
        contract = {
            "execution": {
                key: value
                for key, value in self.checkpoint_contract.items()
                if key != "base_params"
            },
            "runner": (runner_cls.__module__, runner_cls.__qualname__),
            "runner_settings": runner_settings,
            "coin_overrides": self.coin_override_contract,
            "needed_metrics": sorted(self.needed_metrics),
            "couple_unstuck_emas": self.couple_unstuck_emas,
            "batch_size": self.batch_size,
            "max_dispatch_candidate_bars": self.max_dispatch_candidate_bars,
        }
        return (
            id(self.data),
            self.run,
            json.dumps(contract, sort_keys=True),
        )

    def materialize_suite_candidates(self, candidates: list[dict]) -> list[dict]:
        """Resolve scenario defaults before executing on a compatible runner."""
        side = self.sides[0]
        matrix = self._parameter_matrix(candidates, side)
        result = []
        for candidate, row in zip(candidates, matrix):
            expanded = dict(candidate)
            expanded.update(
                (key if key in GLOBAL_RISK_PARAM_KEYS else f"{side}_{key}", float(value))
                for key, value in zip(self.param_keys, row)
            )
            # Metric normalization also reads defaults for the disabled side.
            for pside in ("long", "short"):
                expanded.setdefault(
                    f"{pside}_total_wallet_exposure_limit",
                    self.base_total_wallet_exposure_limits[pside],
                )
                expanded.setdefault(
                    f"{pside}_n_positions", self.base_n_positions[pside]
                )
            result.append(expanded)
        return result

    def evaluate(self, candidates: list[dict]) -> list[dict]:
        return self._evaluate(candidates, include_status=False)

    def evaluate_results(self, candidates: list[dict]):
        """Preserve actual simulator terminal status for authoritative consumers."""
        return self._evaluate(candidates, include_status=True)

    def _evaluate(self, candidates: list[dict], *, include_status: bool):
        results = []
        torch = self._torch
        if getattr(self, "_cuda_residency", None) is not None:
            if not candidates:
                self.last_profile = {}
                return results
            self._cuda_residency.activate(self)
        fused_runner = getattr(self, "fused_runner", None)
        profile_runners = (
            (fused_runner,)
            if fused_runner is not None
            else tuple(self.runners[side] for side in self.sides)
        )
        profile_started = time.perf_counter() if self.profile_enabled else 0.0
        profile = (
            _new_gpu_proxy_profile(
                self,
                candidates,
                profile_runners,
                coin_count=max(
                    (int(getattr(runner, "n_coins", 1)) for runner in profile_runners),
                    default=1,
                ),
                side_count=len(self.sides),
            )
            if self.profile_enabled
            else None
        )
        self.last_profile = {}
        dispatch_batch_size = getattr(self, "dispatch_batch_size", self.batch_size)
        interrupt_check = getattr(self, "interrupt_check", lambda: None)
        progress = _new_gpu_dispatch_progress(
            len(candidates),
            dispatch_batch_size,
            adaptive=False,
        )
        for start, chunk in proxy_batches(self, candidates, dispatch_batch_size):
            chunk_profile_started = (
                time.perf_counter() if profile is not None else 0.0
            )
            interrupt_check()
            stage_started = (
                time.perf_counter() if self.profile_enabled else 0.0
            )
            parameter_matrices = {
                side: self._parameter_matrix(chunk, side) for side in self.sides
            }
            if profile is not None:
                profile["timings_seconds"]["candidate_materialization"] += (
                    time.perf_counter() - stage_started
                )
            ew_metadata = None
            if fused_runner is not None:
                fused_parameters = np.concatenate(
                    (parameter_matrices["long"], parameter_matrices["short"]),
                    axis=1,
                )
                raw_output = fused_runner.run(
                    fused_parameters,
                    profile=self.profile_enabled,
                )
                if profile is not None:
                    _add_gpu_runner_profile(
                        profile,
                        fused_runner,
                        side_count=len(self.sides),
                    )
                interrupt_check()
                stage_started = time.perf_counter() if self.profile_enabled else 0.0
                recovery_distribution = _mps_strategy_eq_recovery_distribution(
                    raw_output, self.needed_metrics
                )
                ew_metrics, ew_metadata = _reduce_exponential_output(
                    raw_output, self.run, self.metrics_data, self.needed_metrics
                )
                raw_output.update(ew_metrics)
                if profile is not None:
                    synchronize()
                    profile["timings_seconds"]["metric_reduction"] += (
                        time.perf_counter() - stage_started
                    )
                    stage_started = time.perf_counter()
                output = {
                    key: value.cpu()
                    for key, value in raw_output.items()
                    if key in CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS | EXPONENTIAL_EQUITY_METRICS
                }
                if recovery_distribution is not None:
                    recovery_distribution = recovery_distribution.cpu()
                if profile is not None:
                    profile["timings_seconds"]["device_to_host"] += (
                        time.perf_counter() - stage_started
                    )
            else:
                raw_side_outputs = {}
                for side in self.sides:
                    raw_side_outputs[side] = self.runners[side].run(
                        parameter_matrices[side],
                        profile=self.profile_enabled,
                    )
                    if profile is not None:
                        _add_gpu_runner_profile(profile, self.runners[side])
                    interrupt_check()
                stage_started = time.perf_counter() if self.profile_enabled else 0.0
                recovery_distribution = (
                    _mps_strategy_eq_recovery_distribution(
                        raw_side_outputs[self.sides[0]], self.needed_metrics
                    )
                    if len(self.sides) == 1
                    else None
                )
                if len(self.sides) == 1:
                    ew_metrics, ew_metadata = _reduce_exponential_output(
                        raw_side_outputs[self.sides[0]], self.run, self.metrics_data, self.needed_metrics
                    )
                    raw_side_outputs[self.sides[0]].update(ew_metrics)
                if profile is not None:
                    synchronize()
                    profile["timings_seconds"]["metric_reduction"] += (
                        time.perf_counter() - stage_started
                    )
                    stage_started = time.perf_counter()
                side_outputs = {
                    side: {
                        key: value.cpu()
                        for key, value in raw_side_outputs[side].items()
                        if key in CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS | EXPONENTIAL_EQUITY_METRICS
                    }
                    for side in self.sides
                }
                if recovery_distribution is not None:
                    recovery_distribution = recovery_distribution.cpu()
                if profile is not None:
                    profile["timings_seconds"]["device_to_host"] += (
                        time.perf_counter() - stage_started
                    )
            stage_started = time.perf_counter() if self.profile_enabled else 0.0
            stage_device_to_host_before = (
                profile["timings_seconds"]["device_to_host"]
                if profile is not None
                else 0.0
            )
            stage_runner_seconds_before = (
                _gpu_profile_runner_seconds(profile["timings_seconds"])
                if profile is not None
                else 0.0
            )
            if fused_runner is None and len(self.sides) == 1:
                side = self.sides[0]
                output = side_outputs[side]
                output.update(
                    _directional_gross_pnl_outputs(
                        side, output["profit_sum"], output["loss_sum"]
                    )
                )
                output.update(
                    _directional_entry_initial_metrics(
                        side, output["entry_initial_balance_pct"]
                    )
                )
            if recovery_distribution is not None:
                output["strategy_eq_recovery_distribution"] = recovery_distribution
            if self.needed_metrics & HARD_STOP_LIFECYCLE_METRICS:
                output["hsl_unified_scope"] = torch.from_numpy(
                    parameter_matrices[self.sides[0]][
                        :, self.param_keys.index("hsl_signal_mode")
                    ] == 0
                )
            timestamp_origin = float(self.metrics_data["ts0"])
            for key in (
                "first_fill_ts",
                "last_fill_ts",
                "last_high_ts",
                "first_eq_ts",
                "last_eq_ts",
            ):
                values = output[key].to(torch.float64)
                output[key] = torch.where(
                    torch.isfinite(values), values + timestamp_origin, values
                )
            if any("_per_exposure_" in name for name in self.needed_metrics):
                output.update(
                    _candidate_wallet_exposure_limit_outputs(
                        chunk,
                        self.base_total_wallet_exposure_limits,
                        torch=torch,
                    )
                )
            if any("_per_position_slot" in name for name in self.needed_metrics):
                output.update(
                    _candidate_position_slot_outputs(
                        chunk,
                        self.base_n_positions,
                        self.base_total_wallet_exposure_limits,
                        torch=torch,
                    )
                )
            if (
                self.needed_metrics & RAW_STRATEGY_DRAWDOWN_CONSUMERS
                and "raw_strategy_day_max_dd" not in output
            ):
                raise RuntimeError("GPU raw strategy-risk summary is missing")
            if self.needed_metrics & RAW_STRATEGY_EQUITY_METRICS and not {
                "raw_strategy_day_end_eq", "raw_strategy_day_min_eq"
            }.issubset(output):
                raise RuntimeError("GPU raw strategy-equity daily summaries are missing")
            if (getattr(self, "native_factual_hsl", False)
                    and any(name.startswith("drawdown_worst_ema_strategy_eq") for name in self.needed_metrics)
                    and "hsl_drawdown_ema_max_portfolio" not in output):
                raise RuntimeError("native HSL EMA maximum requires its observed portfolio payload")
            if (getattr(self, "native_factual_hsl", False)
                    and self.strategy_kind == "trailing_martingale"
                    and self.needed_metrics & ENTRY_INTERVAL_METRICS
                    and "entry_interval_native_metrics" not in output):
                raise RuntimeError("native entry intervals require compact exact metrics")
            if (
                "drawdown_worst_mean_1pct_ema_strategy_eq" in self.needed_metrics
                and (fused_runner is not None or len(self.sides) == 1)
                and "hsl_drawdown_ema_mean_worst_1pct_portfolio" not in output
            ):
                raise RuntimeError("GPU portfolio HSL EMA-tail summary is missing")
            objectives = self._compute_objectives(
                output, self.run, self.metrics_data, needed=self.needed_metrics
            )
            if profile is not None:
                profile["timings_seconds"][
                    "metric_reduction"
                ] += _gpu_profile_unattributed_seconds(
                    profile["timings_seconds"],
                    time.perf_counter() - stage_started,
                    device_to_host_before=stage_device_to_host_before,
                    runner_seconds_before=stage_runner_seconds_before,
                )
                stage_started = time.perf_counter()
            arrays = {
                name: value.detach().cpu().numpy() for name, value in objectives.items()
            }
            rows = [
                {name: float(values[index]) for name, values in arrays.items()}
                for index in range(len(chunk))
            ]
            if include_status:
                from optimization.gpu.executor import ReplayResult
                liquidated = (output["liq_step"] >= 0).tolist()
                # Transfer the compact sidecar once per cohort, rather than
                # copying every candidate's batched metadata for each row.
                if ew_metadata is not None:
                    ew_metadata = _exponential_metadata_to_host(ew_metadata)
                results.extend(
                    ReplayResult(row, bool(flag),
                                 {"ew": _exponential_metadata_row(ew_metadata, index)}
                                 if ew_metadata is not None else None)
                    for index, (row, flag) in enumerate(zip(rows, liquidated))
                )
            else:
                results.extend(rows)
            if profile is not None:
                profile["timings_seconds"]["result_materialization"] += (
                    time.perf_counter() - stage_started
                )
                profile["dispatch_chunk_wall_seconds"].append(
                    time.perf_counter() - chunk_profile_started
                )
            _update_gpu_dispatch_progress(
                progress,
                completed_candidates=start + len(chunk),
                strategy=str(getattr(self, "strategy_kind", "unknown")),
            )
        if profile is not None:
            self.last_profile = _finish_gpu_proxy_profile(profile, profile_started)
        return results


# Compatibility alias retained for callers and tests from the EMA-only slices.
