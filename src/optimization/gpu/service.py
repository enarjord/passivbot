from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import logging
import os
import time

import numpy as np

from config.shared_bot import flatten_shared_bot_side
from optimization.gpu.metric_registry import (
    BTC_INTRADAY_RISK_METRICS,
    ENTRY_INTERVAL_METRICS,
    EQUITY_BALANCE_DIFF_METRICS,
    HARD_STOP_PROXY_METRICS,
)
from optimization.gpu.model import (
    EMA_ANCHOR_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    GPU_STRATEGY_PARAM_KEYS,
    HSL_COIN_OVERRIDE_PATHS,
    MPS_MULTICOIN_MAX_COINS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
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
    build_mps_data,
    build_mps_multicoin_data,
    encode_hsl_panic_order_type,
    encode_tm_retracement_base_pct,
    flatten_trailing_martingale_params,
    gpu_side_enabled,
    validate_hsl_settings,
    validate_hsl_signal_topology,
    validate_single_coin_hsl_signal_topology,
)


CORE_OUTPUT_KEYS = {
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
    "kernel_execution",
    "metric_reduction",
)

_GPU_DISPATCH_PROGRESS_INTERVAL_SECONDS = 30.0


def _new_gpu_dispatch_progress(candidate_count: int, dispatch_batch_size: int):
    if candidate_count <= dispatch_batch_size:
        return None
    started = time.monotonic()
    return {
        "candidate_count": int(candidate_count),
        "dispatch_batch_size": int(dispatch_batch_size),
        "started": started,
        "last_log": started,
        "total_chunks": (
            int(candidate_count) + int(dispatch_batch_size) - 1
        )
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
        and now - float(progress["last_log"])
        < _GPU_DISPATCH_PROGRESS_INTERVAL_SECONDS
    ):
        return
    rate = completed_candidates / max(elapsed, 1.0e-12)
    remaining = int(progress["candidate_count"]) - completed_candidates
    logging.info(
        "GPU proxy dispatch progress | strategy=%s chunks=%d/%d "
        "candidates=%d/%d elapsed=%.1fs eta=%.1fs",
        strategy,
        completed_chunks,
        int(progress["total_chunks"]),
        completed_candidates,
        int(progress["candidate_count"]),
        elapsed,
        remaining / max(rate, 1.0e-12),
    )
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
            bool(getattr(runner, "hsl_ema_tail_enabled", False))
            for runner in runners
        ),
        "hsl_raw_drawdown": any(
            bool(getattr(runner, "hsl_raw_drawdown_enabled", False))
            for runner in runners
        ),
        "hsl_raw_tail": any(
            bool(getattr(runner, "hsl_raw_tail_enabled", False))
            for runner in runners
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
    return sum(
        float(timings.get(key, 0.0))
        for key in _GPU_PROFILE_RUNNER_TIMING_KEYS
    )


def _gpu_profile_unattributed_seconds(
    timings: dict,
    elapsed_seconds: float,
    *,
    device_to_host_before: float,
    runner_seconds_before: float,
) -> float:
    extra_device_to_host = (
        float(timings["device_to_host"]) - float(device_to_host_before)
    )
    extra_runner_seconds = (
        _gpu_profile_runner_seconds(timings) - float(runner_seconds_before)
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
    profile["actual_dispatch_batch_sizes"].append(batch_size)
    dispatch_specialization = runner_profile.get("dispatch_specialization")
    if dispatch_specialization is not None:
        profile["dispatch_specializations"].append(dict(dispatch_specialization))
    profile["dispatch_count"] += dispatch_count
    profile["cold_dispatch_count"] += dispatch_count if cold else 0
    profile["warm_dispatch_count"] += 0 if cold else dispatch_count
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
        candidate_steps = int(
            np.clip(candidate_steps_array, 0, runner_steps).sum()
        )
    profile["kernel_candidate_bars"] += (
        candidate_steps
        * int(getattr(runner, "n_coins", 1))
        * int(side_count)
        * dispatch_count
    )
    timings = profile["timings_seconds"]
    timings["candidate_packing"] += float(
        runner_profile.get("cpu_pack_seconds", 0.0)
    )
    timings["upload_and_buffer_clear"] += float(
        runner_profile.get("upload_and_zero_seconds", 0.0)
    ) + float(runner_profile.get("pre_dispatch_sync_seconds", 0.0))
    compile_seconds = float(runner_profile.get("compile_seconds", 0.0))
    timings["cold_compilation" if cold else "warm_library_lookup"] += (
        compile_seconds
    )
    timings["kernel_execution"] += float(
        runner_profile.get("kernel_seconds", 0.0)
    )
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
    last_eq_values = np.asarray(
        last_eq_ts.detach().cpu().numpy(), dtype=np.float64
    )
    terminal_last_eq = last_eq_values[terminal]
    finite = np.isfinite(terminal_last_eq)
    estimated_steps = np.ones(terminal_count, dtype=np.int64)
    if np.any(finite):
        estimated_steps[finite] = np.rint(
            terminal_last_eq[finite] / max(1, int(interval_ms))
        ).astype(np.int64) - int(effective_start_step)
    effective_step_count = max(
        1, int(effective_end_step) - int(effective_start_step)
    )
    estimated_steps = np.clip(
        estimated_steps, 1, max(1, effective_step_count - 2)
    )
    profile["terminal_candidate_count"] += terminal_count
    profile["terminal_without_equity_count"] += int((~finite).sum())
    profile["estimated_post_terminal_candidate_bars"] += int(
        np.maximum(effective_step_count - 2 - estimated_steps, 0).sum()
    ) * int(profile.get("side_count", 1))
    profile["_terminal_step_fractions"].extend(
        (estimated_steps / max(1, effective_step_count - 2)).tolist()
    )


def _finish_gpu_proxy_profile(profile: dict, started: float) -> dict:
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
    profile["estimated_post_terminal_candidate_bar_fraction"] = (
        avoidable / max(1, int(profile["candidate_bars"]))
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
    "hsl_tier_samples_yellow",
    "hsl_tier_samples_orange",
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
    directional_hsl_rolling_capacity: int | None = None,
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
        "filter_by_min_effective_cost",
        "dynamic_wel_by_tradability",
        "hedge_mode",
        "max_realized_loss_pct",
        "pnls_max_lookback_days",
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
        "version": 1,
        "strategy_kind": str(strategy_kind),
        "exchange": str(exchange),
        "coins": [str(coin) for coin in backtest_params.get("coins", [])],
        "enabled_sides": sorted(str(side) for side in enabled_sides),
        "hlcvs": {
            "shape": [int(value) for value in hlcv_values.shape],
            "dtype": str(hlcv_values.dtype.str),
            "sha256": hashlib.sha256(
                memoryview(hlcv_values).cast("B")
            ).hexdigest(),
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
                str(key): float(value)
                for key, value in sorted((params or {}).items())
            }
            for side, params in sorted((base_params or {}).items())
        },
    }
    if directional_hsl_rolling_capacity is not None:
        contract["directional_hsl_rolling_capacity"] = int(
            directional_hsl_rolling_capacity
        )
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
    safe_batch_size = max(
        1, max_candidate_bars // per_candidate_work
    )
    return min(requested_batch_size, safe_batch_size)


def _single_coin_candle_interval_minutes(backtest_params: dict) -> int:
    raw_interval = backtest_params.get("candle_interval_minutes", 1)
    try:
        interval = float(raw_interval)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MPS single-coin proxy requires candle_interval_minutes to be an "
            "integer >= 1"
        ) from exc
    if not np.isfinite(interval) or interval < 1.0 or not interval.is_integer():
        raise ValueError(
            "MPS single-coin proxy requires candle_interval_minutes to be an "
            "integer >= 1"
        )
    return int(interval)


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
        "GPU MPS dispatch safety cap active | requested_batch=%d dispatch_batch=%d "
        "bars=%d coins=%d sides=%d max_candidate_bars=%d",
        requested_batch_size,
        dispatch_batch_size,
        n_bars,
        n_coins,
        n_sides,
        max_candidate_bars,
    )

_DUAL_SIDE_MULTICOIN_INTRADAY_CUTOFF_METRICS = {
    "adg_pnl",
    "adg_pnl_w",
    "fills_analysis_duration_days",
    "fills_active_days_count",
    "fills_active_days_ratio",
    "fills_active_symbols_count",
    "fills_count",
    "fills_count_close",
    "fills_count_entry",
    "fills_count_long",
    "fills_count_short",
    "fills_entry_per_close",
    "fills_per_day",
    "fills_per_day_close",
    "fills_per_day_entry",
    "fills_per_day_long",
    "fills_per_day_per_position_slot",
    "fills_per_day_per_position_slot_long",
    "fills_per_day_per_position_slot_short",
    "fills_per_day_short",
    "fills_top_symbol_share",
    "long_short_profit_ratio",
    "loss_profit_ratio_long",
    "loss_profit_ratio_short",
    "mdg_pnl",
    "mdg_pnl_w",
    "peak_recovery_days_equity_usd",
    "peak_recovery_hours_equity_usd",
    "peak_recovery_days_pnl",
    "peak_recovery_hours_pnl",
    "position_held_days_mean",
    "position_held_hours_mean",
    "positions_held_per_day",
    "pnl_ratio_long_short",
    "sharpe_ratio_pnl",
    "sharpe_ratio_pnl_w",
    "sortino_ratio_pnl",
    "sortino_ratio_pnl_w",
    "volume_pct_per_day_avg_w",
}

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
        "btc_analysis": any(
            _metric_uses_btc_analysis(metric) for metric in metrics
        ),
        "btc_intraday_risk": bool(metrics & BTC_INTRADAY_RISK_METRICS),
        "equity_balance_diff": bool(metrics & EQUITY_BALANCE_DIFF_METRICS),
        "entry_interval": bool(
            str(strategy_kind).strip().lower() == "trailing_martingale"
            and metrics & ENTRY_INTERVAL_METRICS
        ),
        "strategy_eq_recovery_distribution": bool(
            metrics & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS
        ),
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


def _mps_strategy_eq_recovery_distribution(output: dict, needed_metrics):
    """Run the opt-in recovery postprocessor before proxy outputs leave MPS."""

    if not set(needed_metrics) & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS:
        return None
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


def _directional_coin_hsl_lookback_bars(
    backtest_params: dict,
    *,
    signal_mode: str,
    hsl_enabled: bool,
) -> int:
    """Translate Rust's finite coin-HSL PnL window into candle bars."""

    if not hsl_enabled or str(signal_mode).strip().lower() != "coin":
        return 0
    lookback_days = float(backtest_params.get("pnls_max_lookback_days", -1.0))
    if lookback_days < 0.0:
        return 0
    interval_minutes = int(backtest_params["candle_interval_minutes"])
    return max(
        1,
        int(np.ceil(lookback_days * 24.0 * 60.0 / interval_minutes)),
    )


def _require_multicoin_metric_topology(
    sides, needed_metrics, *, shared_account_controller: bool = False
) -> None:
    unsupported = (
        set(needed_metrics) & _DUAL_SIDE_MULTICOIN_INTRADAY_CUTOFF_METRICS
        if len(sides) == 2 and not shared_account_controller
        else set()
    )
    if unsupported:
        raise ValueError(
            "MPS dual-side multicoin proxy cannot reconstruct the intraday "
            "shared-liquidation cutoff required by these metrics: "
            + ", ".join(sorted(unsupported))
        )


def _multicoin_exposure_eligible_coins(
    per_side_coin_overrides: dict[str, np.ndarray],
    sides,
    wallet_exposure_column: int,
) -> np.ndarray:
    coin_count = next(iter(per_side_coin_overrides.values())).shape[0]
    eligible = np.zeros(coin_count, dtype=bool)
    for side in sides:
        fixed_coin_wels = per_side_coin_overrides[side][
            :, wallet_exposure_column
        ]
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
            n_positions = float(
                candidate.get(n_positions_key, base_n_positions[side])
            )
            limit = float(candidate.get(limit_key, base_limits[side]))
            values.append(
                n_positions if n_positions > 0.0 and limit > 0.0 else 0.0
            )
        outputs[f"position_slots_{side}"] = torch.from_numpy(
            np.asarray(values, dtype=np.float64)
        )
    return outputs


def _single_coin_exposure_params(risk: dict, *, side: str) -> dict[str, float]:
    allowance_mode = str(
        risk.get("we_excess_allowance_mode", "bounded")
    ).strip().lower()
    if allowance_mode not in {"bounded", "legacy_raw"}:
        raise ValueError(
            "MPS proxy requires "
            f"bot.{side}.risk.we_excess_allowance_mode to be bounded or "
            f"legacy_raw, got {allowance_mode!r}"
        )
    return {
        "we_excess_allowance_pct": float(
            risk.get("we_excess_allowance_pct", 0.0) or 0.0
        ),
        "we_excess_allowance_legacy_raw": float(
            allowance_mode == "legacy_raw"
        ),
        "twel_entry_gate_enabled": float(
            bool(risk.get("total_exposure_entry_gate_enabled", True))
        ),
        "twel_enforcer_threshold": float(
            risk.get("total_exposure_enforcer_threshold", 1.0) or 0.0
        ),
    }


def _position_exposure_enforcer_params(
    risk: dict, *, side: str
) -> dict[str, float]:
    enabled = bool(
        risk.get(
            "position_exposure_enforcer_enabled",
            risk.get("risk_wel_enforcer_enabled", False),
        )
    )
    threshold = float(
        risk.get(
            "position_exposure_enforcer_threshold",
            risk.get("risk_wel_enforcer_threshold", 0.0),
        )
        or 0.0
    )
    if enabled and (not np.isfinite(threshold) or threshold <= 0.0):
        raise ValueError(
            "MPS proxy requires a finite positive "
            f"bot.{side}.risk.position_exposure_enforcer_threshold when the "
            "position exposure enforcer is enabled"
        )
    return {
        "wel_enforcer_enabled": float(enabled),
        "wel_enforcer_threshold": threshold,
    }


def _total_exposure_enforcer_params(
    risk: dict, *, side: str
) -> dict[str, float]:
    policy = str(
        risk.get("total_exposure_enforcer_policy", "reduce_overweight")
    ).strip().lower()
    if policy not in {"reduce_overweight", "reduce_portfolio"}:
        raise ValueError(
            "MPS proxy requires "
            f"bot.{side}.risk.total_exposure_enforcer_policy to be "
            f"reduce_overweight or reduce_portfolio, got {policy!r}"
        )
    return {
        "twel_enforcer_enabled": float(
            bool(risk.get("total_exposure_enforcer_enabled", False))
        ),
        "twel_enforcer_reduce_portfolio": float(policy == "reduce_portfolio"),
    }


def _unstuck_params(bot: dict) -> dict[str, float]:
    return {
        "unstuck_enabled": float(bool(bot["unstuck_enabled"])),
        "unstuck_ema_gating_enabled": float(
            bool(bot["unstuck_ema_gating_enabled"])
        ),
        "unstuck_close_pct": float(bot["unstuck_close_pct"]),
        "unstuck_ema_dist": float(bot["unstuck_ema_dist"]),
        "unstuck_loss_allowance_pct": float(bot["unstuck_loss_allowance_pct"]),
        "unstuck_threshold": float(bot["unstuck_threshold"]),
    }


def _hsl_params(bot: dict, *, signal_mode: str) -> dict[str, float]:
    restart_policy_ids = {"always": 0.0, "threshold": 1.0, "never": 2.0}
    signal_mode = str(signal_mode).strip().lower()
    validate_single_coin_hsl_signal_topology(signal_mode, enabled_side_count=1)
    signal_mode_ids = {"unified": 0.0, "pside": 1.0, "coin": 2.0}
    tier_ratios = bot.get("hsl_tier_ratios", {})
    if isinstance(tier_ratios, dict):
        tier_ratios = dict(tier_ratios)
        if "hsl_tier_ratio_yellow" in bot:
            tier_ratios["yellow"] = bot["hsl_tier_ratio_yellow"]
        if "hsl_tier_ratio_orange" in bot:
            tier_ratios["orange"] = bot["hsl_tier_ratio_orange"]
    validated = validate_hsl_settings(
        {
            "enabled": bot.get("hsl_enabled", False),
            "red_threshold": bot.get("hsl_red_threshold", 0.15),
            "ema_span_minutes": bot.get("hsl_ema_span_minutes", 720.0),
            "cooldown_minutes_after_red": bot.get(
                "hsl_cooldown_minutes_after_red", 0.0
            ),
            "no_restart_drawdown_threshold": bot.get(
                "hsl_no_restart_drawdown_threshold", 1.0
            ),
            "restart_after_red_policy": bot.get(
                "hsl_restart_after_red_policy", "threshold"
            ),
            "tier_ratios": tier_ratios,
            "orange_tier_mode": bot.get(
                "hsl_orange_tier_mode",
                "tp_only_with_active_entry_cancellation",
            ),
            "panic_close_order_type": bot.get(
                "hsl_panic_close_order_type", "limit"
            ),
        },
        field_name="MPS HSL",
    )
    float32_below_one = float(
        np.nextafter(np.float32(1.0), np.float32(0.0))
    )
    for key, value in (
        ("hsl_red_threshold", validated["red_threshold"]),
        (
            "hsl_no_restart_drawdown_threshold",
            validated["no_restart_drawdown_threshold"],
        ),
    ):
        if float32_below_one < value < 1.0:
            raise ValueError(
                f"MPS HSL cannot represent {key}={value} distinctly from 1.0; "
                f"use <= {float32_below_one} or exactly 1.0"
            )
    return {
        "hsl_enabled": float(validated["enabled"]),
        "hsl_red_threshold": validated["red_threshold"],
        "hsl_ema_span_minutes": validated["ema_span_minutes"],
        "hsl_cooldown_minutes_after_red": validated[
            "cooldown_minutes_after_red"
        ],
        "hsl_no_restart_drawdown_threshold": validated[
            "no_restart_drawdown_threshold"
        ],
        "hsl_restart_policy": restart_policy_ids[
            validated["restart_after_red_policy"]
        ],
        "hsl_tier_ratio_yellow": validated["tier_ratios"]["yellow"],
        "hsl_tier_ratio_orange": validated["tier_ratios"]["orange"],
        "hsl_orange_graceful_stop": float(
            validated["orange_tier_mode"] == "graceful_stop"
        ),
        "hsl_signal_mode": signal_mode_ids[signal_mode],
        # Multi-coin kernels replace this initial value with the dynamic
        # effective slot count before each per-coin HSL sample. The value is
        # inert for unified/pside and exact for single-coin coin mode.
        "hsl_slot_count": 1.0,
    }


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
    for coin, (first_valid_idx, last_valid_idx) in enumerate(
        zip(starts, tails)
    ):
        # The exact payload uses this sentinel for a prepared coin with no
        # valid candles.  It contributes neither coverage nor a forced-delist
        # surface, but it remains present in the coin-indexed tensors.
        if (
            first_valid_idx == candle_count
            and last_valid_idx == candle_count - 1
        ):
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
    actual_valid = np.all(
        np.isfinite(packed_hlc) & (packed_hlc > 0.0), axis=2
    )
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
        (candle_indices >= np.asarray(starts, dtype=np.int64)[None, :])
        & (candle_indices <= np.asarray(tails, dtype=np.int64)[None, :])
    )
    actual_coverage = np.any(within_declared_window & actual_valid, axis=1)
    missing = np.flatnonzero(
        np.any(within_declared_window, axis=1) & ~actual_coverage
    )
    for candle_index in missing:
        declared = within_declared_window[candle_index]
        raw_hlc = np.asarray(values[candle_index, declared, :3], dtype=np.float64)
        if bool(np.all(np.isnan(raw_hlc))):
            continue
        raise ValueError(
            "GPU multicoin proxy only supports an all-invalid internal candle "
            "when every declared coin's raw H/L/C is NaN; infinities, finite but "
            "non-positive or float32-unrepresentable values remain fail-closed; "
            f"candle_index={int(candle_index)}, valid_windows={windows}"
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
    """Allow exact balance-only gaps, but reject finite unmodeled prices."""

    values = np.asarray(hlcvs)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(
            "MPS proxy requires HLCVs shaped [time, coin, fields]"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        packed = np.asarray(values[:, :, :3], dtype=np.float32)
    for coin in range(values.shape[1]):
        if not bool(exposure_eligible_coins[coin]):
            continue
        first = max(0, int(first_valid_indices[coin]))
        last = min(int(last_valid_indices[coin]), values.shape[0] - 1)
        if first > last:
            continue
        raw_hlc = np.asarray(values[first : last + 1, coin, :3], dtype=np.float64)
        packed_hlc = packed[first : last + 1, coin]
        packed_valid = (
            np.all(np.isfinite(packed_hlc), axis=1)
            & (packed_hlc[:, 2] > 0.0)
        )
        if require_positive_high_low:
            packed_valid &= (packed_hlc[:, 0] > 0.0) & (
                packed_hlc[:, 1] > 0.0
            )
        exact_balance_only = np.all(np.isnan(raw_hlc), axis=1)
        unsafe = np.flatnonzero(~(packed_valid | exact_balance_only))
        if unsafe.size:
            candle_index = first + int(unsafe[0])
            raise ValueError(
                "MPS proxy only supports internal "
                "invalid candles whose raw H/L/C values are all NaN; "
                "infinite, finite but non-positive, partially invalid, or float32-"
                "unrepresentable prices remain fail-closed; "
                f"coin index {coin}, invalid candle at {candle_index}"
            )


def _require_no_unsafe_single_coin_candles(
    hlcvs,
    *,
    first_valid_idx: int,
    last_valid_idx: int,
) -> None:
    values = np.asarray(hlcvs)
    if (
        values.ndim == 3
        and values.shape[1] > 0
        and values.shape[2] >= 3
        and 0 <= int(first_valid_idx) < values.shape[0]
        and bool(np.all(np.isnan(values[int(first_valid_idx), 0, :3])))
    ):
        raise ValueError(
            "MPS single-coin proxy requires the first-valid candle to have "
            f"finite positive H/L/C; first_valid_idx={first_valid_idx}"
        )
    if (
        values.ndim == 3
        and values.shape[1] > 0
        and values.shape[2] >= 3
        and 0 <= int(last_valid_idx) < values.shape[0]
        and int(last_valid_idx) + 1400 < values.shape[0]
        and bool(np.all(np.isnan(values[int(last_valid_idx), 0, :3])))
    ):
        raise ValueError(
            "MPS single-coin proxy requires the forced-delist final candle "
            f"to have finite positive H/L/C; last_valid_idx={last_valid_idx}"
        )
    _require_exact_safe_proxy_candles(
        values,
        exposure_eligible_coins=[True],
        first_valid_indices=[first_valid_idx],
        last_valid_indices=[last_valid_idx],
        require_positive_high_low=True,
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


def _prepared_single_coin_side_enabled(config: dict, side: str, bot: dict) -> bool:
    """Match exact Rust eligibility for the one coin prepared by the payload."""

    if "entry_eligible" not in bot:
        raise ValueError(
            f"GPU single-coin payload for {side} is missing entry_eligible"
        )
    return gpu_side_enabled(config, side) and bool(bot["entry_eligible"])


def _combine_hedged_multicoin_hsl_outputs(long: dict, short: dict) -> dict:
    """Reduce two pside HSL summaries without inventing shared episode state.

    Lifecycle and panic aggregates have the same sum/max/count reductions as
    exact Rust. Minute-level max-tier overlap is not recoverable from two
    histograms; the conservative severity-first allocation below is retained
    only so unrelated lifecycle metrics can share the normal reducer. The
    optimizer gate rejects all three time-in-tier metrics for this topology.
    """

    combined = {
        "hsl_long_enabled": (long["hsl_long_enabled"] > 0)
        | (short["hsl_long_enabled"] > 0),
        "hsl_short_enabled": (long["hsl_short_enabled"] > 0)
        | (short["hsl_short_enabled"] > 0),
    }
    for key in (
        "hsl_triggers_long",
        "hsl_triggers_short",
        "hsl_restarts_long",
        "hsl_restarts_short",
        "hsl_duration_sum_steps",
        "hsl_duration_count",
        "hsl_trigger_drawdown_sum",
        "hsl_trigger_drawdown_count",
        "hsl_flatten_time_sum_steps",
        "hsl_flatten_time_count",
        "hsl_restart_retrigger_count",
        "hsl_halt_to_restart_equity_loss",
        "hsl_panic_close_loss_sum",
        "hsl_panic_loss_drawdown_sum",
        "hsl_panic_loss_drawdown_count",
    ):
        combined[key] = long[key] + short[key]
    for key in (
        "hsl_duration_max_steps",
        "hsl_panic_close_loss_max",
        "hsl_panic_loss_drawdown_max",
        "hsl_drawdown_ema_max_long",
        "hsl_drawdown_ema_max_short",
        "hsl_drawdown_ema_mean_worst_1pct_long",
        "hsl_drawdown_ema_mean_worst_1pct_short",
        "hsl_drawdown_raw_max_long",
        "hsl_drawdown_raw_max_short",
        "hsl_drawdown_raw_mean_worst_1pct_long",
        "hsl_drawdown_raw_mean_worst_1pct_short",
        "hsl_strategy_eq_recovery_max_ms_long",
        "hsl_strategy_eq_recovery_max_ms_short",
    ):
        combined[key] = long[key].maximum(short[key])

    long_count = long["hsl_panic_loss_drawdown_count"]
    short_count = short["hsl_panic_loss_drawdown_count"]
    long_has = long_count > 0.0
    short_has = short_count > 0.0
    both_have = long_has & short_has
    zeros = long_count.new_zeros(long_count.shape)
    combined["hsl_panic_loss_drawdown_min"] = long[
        "hsl_panic_loss_drawdown_min"
    ].minimum(short["hsl_panic_loss_drawdown_min"]).where(
        both_have,
        long["hsl_panic_loss_drawdown_min"].where(
            long_has,
            short["hsl_panic_loss_drawdown_min"].where(short_has, zeros),
        ),
    )

    total = long["hsl_tier_samples_total"].maximum(
        short["hsl_tier_samples_total"]
    )
    red = (long["hsl_tier_samples_red"] + short["hsl_tier_samples_red"]).minimum(
        total
    )
    remaining = (total - red).clamp(min=0.0)
    orange = (
        long["hsl_tier_samples_orange"] + short["hsl_tier_samples_orange"]
    ).minimum(remaining)
    remaining = (remaining - orange).clamp(min=0.0)
    yellow = (
        long["hsl_tier_samples_yellow"] + short["hsl_tier_samples_yellow"]
    ).minimum(remaining)
    combined.update(
        {
            "hsl_tier_samples_total": total,
            "hsl_tier_samples_yellow": yellow,
            "hsl_tier_samples_orange": orange,
            "hsl_tier_samples_red": red,
        }
    )
    return combined


def _refresh_hedged_multicoin_hsl_at_portfolio_cutoff(
    *,
    side_outputs: dict,
    runners: dict,
    parameter_matrices: dict,
    combined_output: dict,
    start_minute_of_day: int,
    interrupt_check=None,
    profile: bool = False,
    runner_profile_callback=None,
    profile_timings: dict | None = None,
) -> bool:
    """Replace full-run directional HSL summaries at a portfolio cutoff.

    The conservative hedged reducer may stop before either isolated side. Re-run
    only those candidates through the last complete pre-liquidation day so
    scalar HSL events after the combined coverage boundary cannot leak into
    proxy objectives or limits.
    """

    cutoff_days = combined_output["liq_step"]
    cutoff_mask = cutoff_days >= 0
    if not bool(cutoff_mask.any().item()):
        return False
    indices = np.flatnonzero(cutoff_mask.cpu().numpy())
    end_steps = (
        np.rint(cutoff_days[cutoff_mask].cpu().numpy()).astype(np.int64) * 1440
        - int(start_minute_of_day)
    )
    end_steps = np.maximum(end_steps, 1).astype(np.int32)
    interrupt_check = interrupt_check or (lambda: None)
    for side in ("long", "short"):
        interrupt_check()
        run_kwargs = {"end_steps": end_steps}
        if profile:
            run_kwargs["profile"] = True
        truncated = runners[side].run(
            parameter_matrices[side][indices], **run_kwargs
        )
        if runner_profile_callback is not None:
            runner_profile_callback(
                runners[side], effective_candidate_steps=end_steps
            )
        interrupt_check()
        transfer_started = time.perf_counter() if profile else 0.0
        for key in DIRECTIONAL_HSL_OUTPUT_KEYS:
            side_outputs[side][key][cutoff_mask] = truncated[key].cpu()
        if profile_timings is not None:
            profile_timings["device_to_host"] += (
                time.perf_counter() - transfer_started
            )
    return True


def _combine_hedged_multicoin_outputs(
    long: dict,
    short: dict,
    starting_balance: float,
    liquidation_threshold: float,
    start_minute_of_day: int,
    interval_ms: int,
):
    """Build a conservative portfolio surface from independent directional screens.

    This is deliberately only a ranking proxy. The unchanged Rust backtest remains
    authoritative for every accepted result and the optimizer's drift gates halt on
    material disagreement.
    """

    active = long["day_min_eq"].isfinite() & short["day_min_eq"].isfinite()
    day_count = int(active.shape[1])
    day_ids = active.new_tensor(range(day_count), dtype=long["liq_step"].dtype)
    no_liquidation = long["liq_step"].new_full(long["liq_step"].shape, day_count)
    directional_liquidation_day = long["liq_step"].where(
        long["liq_step"] >= 0, no_liquidation
    ).minimum(
        short["liq_step"].where(short["liq_step"] >= 0, no_liquidation)
    )
    raw_combined_min = (
        long["day_min_eq"] + short["day_min_eq"] - float(starting_balance)
    )
    raw_combined_min_balance = (
        long["day_min_balance"]
        + short["day_min_balance"]
        - float(starting_balance)
    )
    portfolio_floor = max(0.0, float(starting_balance)) * max(
        0.0, float(liquidation_threshold)
    )
    portfolio_breach = active & (
        (raw_combined_min <= portfolio_floor) | (raw_combined_min_balance <= 0.0)
    )
    portfolio_liquidation_day = portfolio_breach.to(
        dtype=long["liq_step"].dtype
    ).argmax(dim=1).where(
        portfolio_breach.any(dim=1), no_liquidation
    )
    terminal_day = directional_liquidation_day.minimum(portfolio_liquidation_day)
    liquidated = terminal_day < day_count
    liquidation_day = terminal_day.where(
        liquidated, -no_liquidation.new_ones(())
    )
    active &= (~liquidated).unsqueeze(1) | (
        day_ids.unsqueeze(0) < terminal_day.unsqueeze(1)
    )

    combined = {}
    for key in ("day_end_eq", "day_min_eq"):
        values = raw_combined_min if key == "day_min_eq" else (
            long[key] + short[key] - float(starting_balance)
        )
        if key == "day_min_eq":
            values = values.where(active, values.new_full((), float("inf")))
        else:
            values = values.where(active, values.new_zeros(()))
        combined[key] = values

    combined["day_max_dd"] = (
        long["day_max_dd"] + short["day_max_dd"]
    ).clamp(max=1.0).where(active, long["day_max_dd"].new_zeros(()))
    combined["day_volume"] = (long["day_volume"] + short["day_volume"]).where(
        active, long["day_volume"].new_zeros(())
    )
    combined["day_has_fill"] = (
        long["day_has_fill"] | short["day_has_fill"]
    ) & active
    combined["day_net_pnl"] = (
        long["day_net_pnl"] + short["day_net_pnl"]
    ).where(active, long["day_net_pnl"].new_zeros(()))
    combined["day_last_fill_balance"] = (
        long["day_last_fill_balance"]
        + short["day_last_fill_balance"]
        - float(starting_balance)
    ).where(active, long["day_last_fill_balance"].new_zeros(()))
    combined["day_fill_count"] = (
        long["day_fill_count"] + short["day_fill_count"]
    ).where(active, long["day_fill_count"].new_zeros(()))
    combined["max_dd"] = (long["max_dd"] + short["max_dd"]).clamp(max=1.0)
    combined["held_max_ms"] = long["held_max_ms"].maximum(short["held_max_ms"])
    combined["held_sum_ms"] = long["held_sum_ms"] + short["held_sum_ms"]
    combined["held_count"] = long["held_count"] + short["held_count"]
    combined["position_unchanged_max_ms"] = long[
        "position_unchanged_max_ms"
    ].maximum(short["position_unchanged_max_ms"])
    combined["gap_hist"] = long["gap_hist"] + short["gap_hist"]
    combined["gap_max_ms"] = long["gap_max_ms"].maximum(short["gap_max_ms"])
    combined["first_fill_ts"] = _nan_min(
        long["first_fill_ts"], short["first_fill_ts"]
    )
    combined["last_fill_ts"] = _nan_max(
        long["last_fill_ts"], short["last_fill_ts"]
    )
    combined["recovery_max_ms"] = long["recovery_max_ms"].maximum(
        short["recovery_max_ms"]
    )
    # The earlier directional high produces the longer, safer final-recovery estimate.
    combined["last_high_ts"] = _nan_min(
        long["last_high_ts"], short["last_high_ts"]
    )
    combined["first_eq_ts"] = _nan_max(
        long["first_eq_ts"], short["first_eq_ts"]
    )
    last_eq_ts = _nan_min(long["last_eq_ts"], short["last_eq_ts"])
    # Daily summaries do not reveal the exact intra-day portfolio breach. Stop at
    # the final complete candle before that UTC day so completion cannot imply
    # coverage beyond the conservative combined-equity liquidation surface.
    terminal_day_start_ms = (
        terminal_day.to(last_eq_ts.dtype) * 86_400_000.0
        - float(start_minute_of_day) * 60_000.0
    ).clamp(min=0.0)
    complete_tail_ms = (terminal_day_start_ms - float(interval_ms)).clamp(min=0.0)
    first_eq_ts = combined["first_eq_ts"]
    complete_tail_ms = complete_tail_ms.maximum(first_eq_ts).where(
        first_eq_ts.isfinite(), complete_tail_ms
    )
    combined["last_eq_ts"] = last_eq_ts.minimum(complete_tail_ms).where(
        liquidated, last_eq_ts
    )

    combined["liq_step"] = liquidation_day
    combined["profit_sum"] = long["profit_sum"] + short["profit_sum"]
    combined["loss_sum"] = long["loss_sum"] + short["loss_sum"]
    combined["profit_sum_long"] = long["profit_sum"]
    combined["loss_sum_long"] = long["loss_sum"]
    combined["profit_sum_short"] = short["profit_sum"]
    combined["loss_sum_short"] = short["loss_sum"]
    combined["fill_count"] = long["fill_count"] + short["fill_count"]
    combined["fill_count_entry"] = (
        long["fill_count_entry"] + short["fill_count_entry"]
    )
    combined["fill_count_long"] = (
        long["fill_count_long"] + short["fill_count_long"]
    )
    # Preserve the core output shape for unrelated dual-side metrics. Requests
    # for active-day metrics fail closed before this conservative placeholder
    # can be consumed because overlapping directional buckets need a true union.
    combined["fills_active_days_count"] = long[
        "fills_active_days_count"
    ].maximum(short["fills_active_days_count"])
    combined["pnl_recovery_max_ms"] = long["pnl_recovery_max_ms"].maximum(
        short["pnl_recovery_max_ms"]
    )
    if DIRECTIONAL_HSL_OUTPUT_KEYS <= long.keys() and (
        DIRECTIONAL_HSL_OUTPUT_KEYS <= short.keys()
    ):
        combined.update(_combine_hedged_multicoin_hsl_outputs(long, short))
    return combined


class MpsSingleCoinProxy:
    """Batched directional screening proxy for supported single-coin strategies."""

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
    ):
        try:
            import torch
        except (
            ModuleNotFoundError
        ) as exc:  # pragma: no cover - optional dependency path
            raise ModuleNotFoundError(
                "GPU optimization requires the optional 'gpu-mps' dependencies; "
                "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "GPU optimization requested but Apple MPS is unavailable in this process"
            )

        from optimization.gpu.metrics import (
            BTC_INTRADAY_RISK_METRICS,
            ENTRY_INTERVAL_METRICS,
            EQUITY_BALANCE_DIFF_METRICS,
            compute_objectives,
            validate_gpu_metric_names,
        )
        self.needed_metrics = set(validate_gpu_metric_names(needed_metrics))

        from backtest import build_backtest_payload
        from optimization.gpu.mps_kernel import (
            MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY,
            MpsEmaAnchorRunner,
            MpsTrailingMartingaleRunner,
        )

        self._torch = torch
        self._compute_objectives = compute_objectives
        self.btc_analysis_enabled = any(
            _metric_uses_btc_analysis(metric) for metric in self.needed_metrics
        )
        self.btc_risk_enabled = bool(
            self.needed_metrics & BTC_INTRADAY_RISK_METRICS
        )
        self.equity_balance_diff_enabled = bool(
            self.needed_metrics & EQUITY_BALANCE_DIFF_METRICS
        )
        btc_values = np.ascontiguousarray(
            np.asarray(btc, dtype=np.float64).reshape(-1)
        )
        self.batch_size = max(1, int(batch_size))
        self.max_dispatch_candidate_bars = int(max_dispatch_candidate_bars)
        self.interrupt_check = interrupt_check or (lambda: None)
        self.profile_enabled = os.environ.get(
            "PASSIVBOT_GPU_PROFILE", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.last_profile: dict = {}
        self.strategy_kind = str(
            config.get("live", {}).get("strategy_kind", "")
        ).strip().lower()
        self.entry_interval_enabled = bool(
            self.strategy_kind == "trailing_martingale"
            and self.needed_metrics & ENTRY_INTERVAL_METRICS
        )
        if self.strategy_kind not in GPU_STRATEGY_PARAM_KEYS:
            raise ValueError(
                "MPS single-coin proxy supports ema_anchor or "
                f"trailing_martingale, got {self.strategy_kind!r}"
            )
        self.param_keys = GPU_STRATEGY_PARAM_KEYS[self.strategy_kind]

        payload = build_backtest_payload(
            np.ascontiguousarray(hlcvs),
            mss,
            copy.deepcopy(config),
            exchange,
            btc_values,
            timestamps,
            metrics_only=True,
            skip_btc_analysis=not self.btc_analysis_enabled,
        )
        if len(payload.bot_params_list) != 1:
            raise ValueError(
                "GPU foundation supports exactly one backtest coin; "
                f"prepared {len(payload.bot_params_list)}"
            )
        backtest_params = payload.backtest_params
        candle_interval_minutes = _single_coin_candle_interval_minutes(
            backtest_params
        )
        long_bot = payload.bot_params_list[0]["long"]
        short_bot = payload.bot_params_list[0]["short"]
        self.enabled = {
            side: _prepared_single_coin_side_enabled(config, side, bot)
            for side, bot in (("long", long_bot), ("short", short_bot))
        }
        if not any(self.enabled.values()):
            raise ValueError("GPU foundation requires at least one enabled side")
        enabled_side_count = sum(self.enabled.values())
        self.dispatch_batch_size = _mps_dispatch_batch_size(
            self.batch_size,
            n_bars=len(hlcvs),
            n_sides=enabled_side_count,
            max_candidate_bars=self.max_dispatch_candidate_bars,
        )
        _log_mps_dispatch_cap(
            requested_batch_size=self.batch_size,
            dispatch_batch_size=self.dispatch_batch_size,
            n_bars=len(hlcvs),
            n_coins=1,
            n_sides=enabled_side_count,
            max_candidate_bars=self.max_dispatch_candidate_bars,
        )
        hsl_enabled_sides = [
            side
            for side, bot in (("long", long_bot), ("short", short_bot))
            if self.enabled[side] and bool(bot.get("hsl_enabled"))
        ]
        signal_mode = (
            backtest_params.get("equity_hard_stop_loss", {})
            .get("signal_mode", "unified")
        )
        if hsl_enabled_sides:
            validate_single_coin_hsl_signal_topology(
                signal_mode, enabled_side_count=sum(self.enabled.values())
            )
        hsl_panic_market = {}
        self.base_params = {}
        configured_total_wallet_exposure_limits = {
            side: float(bot["total_wallet_exposure_limit"])
            for side, bot in (("long", long_bot), ("short", short_bot))
        }
        for side, bot in (("long", long_bot), ("short", short_bot)):
            panic_close_order_type = str(
                bot.get("hsl_panic_close_order_type", "limit")
            ).strip().lower()
            if panic_close_order_type not in {"limit", "market"}:
                raise ValueError(
                    f"MPS single-coin HSL requires bot.{side}.hsl."
                    "panic_close_order_type to be limit or market, got "
                    f"{panic_close_order_type!r}"
                )
            hsl_panic_market[side] = (
                self.enabled[side]
                and bool(bot.get("hsl_enabled"))
                and panic_close_order_type == "market"
            )
            strategy = dict(payload.strategy_params_list[0][side])
            risk = config["bot"][side]["risk"]
            if self.strategy_kind == "trailing_martingale":
                strategy = flatten_trailing_martingale_params(strategy, risk)
            else:
                strategy["entry_cooldown_minutes"] = float(
                    risk.get("entry_cooldown_minutes", 0.0) or 0.0
                )
                strategy["total_wallet_exposure_limit"] = float(
                    risk["total_wallet_exposure_limit"]
                )
            strategy.update(_single_coin_exposure_params(risk, side=side))
            if self.strategy_kind == "trailing_martingale":
                strategy.update(
                    _position_exposure_enforcer_params(risk, side=side)
                )
            if self.strategy_kind in {"ema_anchor", "trailing_martingale"}:
                strategy.update(
                    _total_exposure_enforcer_params(risk, side=side)
                )
                strategy.update(_unstuck_params(bot))
                strategy.update(_hsl_params(bot, signal_mode=signal_mode))
            strategy["wallet_exposure_limit"] = float(
                bot.get("wallet_exposure_limit", -1.0)
            )
            missing = [key for key in self.param_keys if key not in strategy]
            if missing:
                raise ValueError(
                    f"GPU {self.strategy_kind} payload for {side} is missing "
                    f"parameters: {missing}"
                )
            self.base_params[side] = strategy

        coins = list(backtest_params.get("coins") or [])
        if len(coins) != 1:
            raise ValueError(
                "GPU single-coin payload coin identity disagrees with prepared "
                f"data: coins={coins}"
            )
        self.static_coin_override_params = {}
        per_side_override_contracts = {}
        for side in ("long", "short"):
            values, contract = _build_single_coin_override_params(
                config=config,
                mss=mss,
                exchange=exchange,
                coin=coins[0],
                payload=payload,
                side=side,
                strategy_kind=self.strategy_kind,
            )
            self.static_coin_override_params[side] = values
            self.base_params[side].update(values)
            per_side_override_contracts[side] = contract
        self.coin_override_contract = {
            "exchange": exchange,
            "coins": coins,
            "sides": [side for side, enabled in self.enabled.items() if enabled],
            "values_by_side": {
                side: per_side_override_contracts[side]["values"]
                for side in ("long", "short")
            },
            "exact_overrides_by_side": {
                side: per_side_override_contracts[side]["exact_overrides"]
                for side in ("long", "short")
            },
            "proxy_mode": "single-coin-exact-last-v1",
        }
        pnl_lookback_bars = _directional_coin_hsl_lookback_bars(
            backtest_params,
            signal_mode=signal_mode,
            hsl_enabled=bool(hsl_enabled_sides),
        )

        self.checkpoint_contract = _gpu_proxy_execution_checkpoint_contract(
            strategy_kind=self.strategy_kind,
            exchange=exchange,
            enabled_sides=[side for side, enabled in self.enabled.items() if enabled],
            hlcvs=hlcvs,
            timestamps=timestamps,
            backtest_params=backtest_params,
            exchange_params=payload.exchange_params,
            base_params=self.base_params,
            btc_prices=btc_values if self.btc_analysis_enabled else None,
            directional_hsl_rolling_capacity=(
                MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY
                if pnl_lookback_bars > 0
                else None
            ),
        )

        self.base_total_wallet_exposure_limits = {
            side: configured_total_wallet_exposure_limits[side]
            for side in ("long", "short")
        }
        self.base_n_positions = {
            side: float(self.enabled[side]) for side in ("long", "short")
        }

        market_params = payload.exchange_params[0]
        self.market = ProxyMarket(
            qty_step=float(market_params["qty_step"]),
            price_step=float(market_params["price_step"]),
            min_qty=float(market_params["min_qty"]),
            min_cost=float(market_params["min_cost"]),
            c_mult=float(market_params["c_mult"]),
            maker_fee=float(market_params["maker_fee"]),
            taker_fee=float(market_params["taker_fee"]),
        )
        interval_ms = candle_interval_minutes * 60_000
        self.run = ProxyRun(
            starting_balance=float(backtest_params["starting_balance"]),
            warmup_bars=max(1, int(backtest_params.get("global_warmup_bars", 0) or 1)),
            trade_start_idx=int(backtest_params["trade_start_indices"][0]),
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
            first_valid_idx=int(backtest_params["first_valid_indices"][0]),
            last_valid_idx=int(backtest_params["last_valid_indices"][0]),
        )
        coin_warmup_minutes = int(
            (backtest_params.get("warmup_minutes") or [0])[0]
        )
        self.history_warmup_bars = max(
            int(self.run.warmup_bars),
            int(np.ceil(coin_warmup_minutes / candle_interval_minutes)),
        )

        high = hlcvs[:, 0, 0].astype(np.float64)
        low = hlcvs[:, 0, 1].astype(np.float64)
        close = hlcvs[:, 0, 2].astype(np.float64)
        _require_no_unsafe_single_coin_candles(
            hlcvs,
            first_valid_idx=self.run.first_valid_idx,
            last_valid_idx=self.run.last_valid_idx,
        )
        if hsl_enabled_sides:
            _require_no_internal_invalid_hsl_candles(
                high,
                low,
                close,
                first_valid_idx=self.run.first_valid_idx,
                last_valid_idx=self.run.last_valid_idx,
            )
        self.data = build_mps_data(high, low, close, timestamps, self.run, self.market)
        self.metrics_data = {
            "ts0": self.data["ts0"],
            "n": self.data["n"],
            "strategy_kind": self.strategy_kind,
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
        runner_cls = (
            MpsTrailingMartingaleRunner
            if self.strategy_kind == "trailing_martingale"
            else MpsEmaAnchorRunner
        )
        runner_kwargs = dict(
            long_enabled=self.enabled["long"],
            short_enabled=self.enabled["short"],
            hedge_mode=bool(backtest_params["hedge_mode"]),
            filter_by_min_effective_cost=bool(
                backtest_params["filter_by_min_effective_cost"]
            ),
            max_realized_loss_pct=float(
                backtest_params.get("max_realized_loss_pct", 1.0)
            ),
            taker_fee=float(market_params["taker_fee"]),
            market_order_slippage_pct=float(
                backtest_params.get("market_order_slippage_pct", 0.0)
            ),
            market_orders_allowed=bool(
                backtest_params.get("market_orders_allowed", False)
            ),
            market_order_near_touch_threshold=float(
                backtest_params.get("market_order_near_touch_threshold", 0.001)
            ),
            hsl_panic_market_long=hsl_panic_market["long"],
            hsl_panic_market_short=hsl_panic_market["short"],
            pnl_lookback_bars=pnl_lookback_bars,
            hsl_ema_tail_enabled=bool(
                self.needed_metrics & _HSL_EMA_TAIL_METRICS
            ),
            hsl_raw_drawdown_enabled=bool(
                self.needed_metrics
                & (_HSL_RAW_DRAWDOWN_METRICS | _HSL_RAW_TAIL_METRICS)
            ),
            hsl_raw_tail_enabled=bool(
                self.needed_metrics & _HSL_RAW_TAIL_METRICS
            ),
            recovery_distribution_enabled=bool(
                self.needed_metrics & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS
            ),
            btc_prices=(
                btc_values
                if self.btc_risk_enabled or self.equity_balance_diff_enabled
                else None
            ),
            btc_risk_enabled=self.btc_risk_enabled,
            equity_balance_diff_enabled=self.equity_balance_diff_enabled,
            entry_interval_enabled=self.entry_interval_enabled,
        )
        if self.strategy_kind == "trailing_martingale":
            runner_kwargs["hsl_diagnostics_enabled"] = _hsl_diagnostics_needed(
                self.needed_metrics
            )
        runner_kwargs["hsl_enabled"] = bool(hsl_enabled_sides)
        self.runner = runner_cls(
            self.market,
            self.run,
            self.data,
            **runner_kwargs,
        )
        shader_topology = getattr(self.runner, "shader_topology", "generic")
        if shader_topology != "generic":
            logging.info(
                "GPU MPS specialized kernel selected | strategy=%s topology=%s",
                self.strategy_kind,
                shader_topology,
            )

    def _parameter_matrix(self, candidates: list[dict]) -> np.ndarray:
        rows = []
        for candidate in candidates:
            row = []
            for side in ("long", "short"):
                merged = dict(self.base_params[side])
                merged.update(
                    {
                        key.removeprefix(f"{side}_"): value
                        for key, value in candidate.items()
                        if key.startswith(f"{side}_")
                    }
                )
                merged.update(
                    getattr(self, "static_coin_override_params", {}).get(side, {})
                )
                row.extend(float(merged[key]) for key in self.param_keys)
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def recent_window_for_history_fraction(
        self, history_fraction: float
    ) -> tuple[int, int]:
        """Map a history fraction to warmup and trade starts for a recent suffix."""

        fraction = float(history_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("GPU history fraction must be finite and in (0, 1]")
        candle_count = int(self.runner.n)
        warmup_readiness_bars = max(
            1,
            int(
                getattr(self, "history_warmup_bars", self.run.warmup_bars)
            ),
        )
        full_trade_start = min(
            candle_count - 3,
            max(2, int(self.run.trade_start_idx), int(self.run.first_valid_idx) + 1),
        )
        suffix_candles = max(
            2,
            int(np.ceil((candle_count - full_trade_start) * fraction)),
        )
        trade_start = max(full_trade_start, candle_count - suffix_candles)
        history_start = max(
            int(self.run.first_valid_idx),
            trade_start - warmup_readiness_bars - 1,
        )
        history_start = min(history_start, trade_start - 1)
        return history_start, trade_start

    def evaluate(
        self,
        candidates: list[dict],
        *,
        end_step: int | None = None,
        history_start_step: int | None = None,
        trade_start_step: int | None = None,
    ) -> list[dict]:
        results: list[dict] = []
        torch = self._torch
        full_candle_count = max(
            3,
            int(
                getattr(
                    self.runner,
                    "n",
                    getattr(self, "metrics_data", {}).get("n", 3),
                )
            ),
        )
        effective_end_step = (
            full_candle_count if end_step is None else int(end_step)
        )
        if not 3 <= effective_end_step <= full_candle_count:
            raise ValueError(
                "GPU single-coin end_step must be between 3 and the full candle "
                f"count {full_candle_count}, got {effective_end_step}"
            )
        bounded_history = (
            history_start_step is not None or trade_start_step is not None
        )
        if bounded_history and (
            history_start_step is None or trade_start_step is None
        ):
            raise ValueError(
                "GPU recent-history evaluation requires both history and trade starts"
            )
        if bounded_history and self.strategy_kind != "trailing_martingale":
            raise ValueError(
                "GPU recent-history evaluation currently requires trailing_martingale"
            )
        effective_history_start = (
            0 if history_start_step is None else int(history_start_step)
        )
        effective_trade_start = (
            int(trade_start_step) if bounded_history else 0
        )
        effective_candle_count = effective_end_step - effective_history_start
        if effective_candle_count < 3:
            raise ValueError("GPU recent-history evaluation requires at least 3 candles")
        side_count = int(bool(getattr(self.runner, "long_enabled", True))) + int(
            bool(getattr(self.runner, "short_enabled", False))
        )
        dispatch_batch_size = (
            int(getattr(self, "dispatch_batch_size", self.batch_size))
            if end_step is None and not bounded_history
            else _mps_dispatch_batch_size(
                self.batch_size,
                n_bars=effective_candle_count,
                n_sides=side_count,
                max_candidate_bars=self.max_dispatch_candidate_bars,
            )
        )
        profile_started = time.perf_counter() if self.profile_enabled else 0.0
        profile = (
            _new_gpu_proxy_profile(
                self,
                candidates,
                (self.runner,),
                coin_count=1,
                side_count=side_count,
                candle_count=effective_candle_count,
                dispatch_batch_size=dispatch_batch_size,
            )
            if self.profile_enabled
            else None
        )
        self.last_profile = {}
        interrupt_check = getattr(self, "interrupt_check", lambda: None)
        progress = _new_gpu_dispatch_progress(
            len(candidates), dispatch_batch_size
        )
        for start in range(0, len(candidates), dispatch_batch_size):
            chunk_profile_started = (
                time.perf_counter() if profile is not None else 0.0
            )
            interrupt_check()
            chunk = candidates[start : start + dispatch_batch_size]
            stage_started = (
                time.perf_counter() if self.profile_enabled else 0.0
            )
            parameter_matrix = self._parameter_matrix(chunk)
            if profile is not None:
                profile["timings_seconds"]["candidate_materialization"] += (
                    time.perf_counter() - stage_started
                )
            runner_kwargs = {
                "profile": self.profile_enabled,
                "end_step": effective_end_step,
            }
            if bounded_history:
                runner_kwargs.update(
                    history_start_step=effective_history_start,
                    trade_start_step=effective_trade_start,
                )
            output = self.runner.run(parameter_matrix, **runner_kwargs)
            if profile is not None:
                _add_gpu_runner_profile(
                    profile,
                    self.runner,
                    side_count=profile["side_count"],
                    effective_candidate_steps=np.full(
                        len(chunk), effective_candle_count, dtype=np.int64
                    ),
                )
            interrupt_check()
            stage_started = (
                time.perf_counter() if self.profile_enabled else 0.0
            )
            recovery_distribution = _mps_strategy_eq_recovery_distribution(
                output, self.needed_metrics
            )
            if profile is not None:
                torch.mps.synchronize()
                profile["timings_seconds"]["metric_reduction"] += (
                    time.perf_counter() - stage_started
                )
                stage_started = time.perf_counter()
            host_output_keys = CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS
            if profile is not None:
                host_output_keys = host_output_keys | {"alive"}
            output = {
                key: value.cpu()
                for key, value in output.items()
                if key in host_output_keys
            }
            if recovery_distribution is not None:
                output["strategy_eq_recovery_distribution"] = (
                    recovery_distribution.cpu()
                )
            if profile is not None:
                profile["timings_seconds"]["device_to_host"] += (
                    time.perf_counter() - stage_started
                )
                stage_started = time.perf_counter()
                _add_gpu_terminal_profile(
                    profile,
                    output,
                    interval_ms=int(self.run.interval_ms),
                    effective_start_step=effective_history_start,
                    effective_end_step=effective_end_step,
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
            metrics_run = self.run
            if bounded_history:
                requested_start_ts_ms = int(
                    self.metrics_data["ts0"]
                    + effective_trade_start * self.run.interval_ms
                )
                metrics_run = replace(
                    self.run,
                    trade_start_idx=effective_trade_start,
                    requested_start_ts_ms=requested_start_ts_ms,
                    guard_ts_ms=requested_start_ts_ms,
                )
            objectives = self._compute_objectives(
                output,
                metrics_run,
                {**self.metrics_data, "n": effective_end_step},
                needed=self.needed_metrics,
            )
            if profile is not None:
                profile["timings_seconds"]["metric_reduction"] += (
                    time.perf_counter() - stage_started
                )
                stage_started = time.perf_counter()
            arrays = {
                name: value.detach().cpu().numpy() for name, value in objectives.items()
            }
            results.extend(
                {name: float(values[index]) for name, values in arrays.items()}
                for index in range(len(chunk))
            )
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
            self.last_profile = _finish_gpu_proxy_profile(
                profile, profile_started
            )
        return results


# Compatibility name retained for downstream imports from the EMA foundation PR.
MpsEmaAnchorProxy = MpsSingleCoinProxy


def _pack_multicoin_hsl_overrides(
    matrix: np.ndarray,
    *,
    row: int,
    start_column: int,
    side_patch: dict,
    effective_bot: dict,
) -> None:
    hsl_patch = side_patch.get("hsl", {}) or {}
    if not hsl_patch:
        return
    packed = _hsl_params(effective_bot, signal_mode="coin")
    missing = object()
    for offset, (key, path) in enumerate(HSL_COIN_OVERRIDE_PATHS):
        value = hsl_patch
        for part in path:
            value = value.get(part, missing) if isinstance(value, dict) else missing
            if value is missing:
                break
        if value is missing:
            continue
        if key == "hsl_panic_market":
            encoded = encode_hsl_panic_order_type(
                value,
                field_name="coin override hsl.panic_close_order_type",
            )
        else:
            encoded = float(packed[key])
        matrix[row, start_column + offset] = encoded


def _build_multicoin_ema_coin_overrides(
    *,
    config: dict,
    mss: dict,
    exchange: str,
    coins: list[str],
    payload,
    side: str,
    resolve_override=None,
) -> tuple[np.ndarray, dict]:
    """Pack exact-last static coin overrides for the Metal EMA proxy."""

    if resolve_override is None:
        from backtest import _get_backtest_coin_override

        resolve_override = _get_backtest_coin_override

    matrix = np.full(
        (len(coins), EMA_ANCHOR_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    exact_overrides = []
    for coin_index, coin in enumerate(coins):
        patch = resolve_override(config, mss, exchange, coin) or {}
        exact_overrides.append(copy.deepcopy(patch))
        side_patch = patch.get("bot", {}).get(side, {})
        strategy_patch = side_patch.get("strategy", {}).get("ema_anchor", {}) or {}
        effective_strategy = payload.strategy_params_list[coin_index][side]
        effective_bot = payload.bot_params_list[coin_index][side]
        for column, key in enumerate(EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS):
            if key in strategy_patch:
                matrix[coin_index, column] = float(effective_strategy[key])
        risk_patch = side_patch.get("risk", {}) or {}
        if "entry_cooldown_minutes" in risk_patch:
            matrix[coin_index, EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN] = float(
                effective_bot.get("risk_entry_cooldown_minutes", 0.0) or 0.0
            )
        # Exact payload construction keeps per-side universe eligibility in
        # entry_eligible and uses a zero WEL sentinel for an ineligible coin.
        # Preserve that sentinel even when no explicit coin override exists so
        # fused long/short proxies may screen different side universes.
        if not bool(effective_bot.get("entry_eligible", True)) or (
            "wallet_exposure_limit" in side_patch
        ):
            matrix[
                coin_index, EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
            ] = float(effective_bot["wallet_exposure_limit"])
        if "we_excess_allowance_pct" in risk_patch:
            matrix[coin_index, EMA_ANCHOR_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN] = float(
                effective_bot.get("risk_we_excess_allowance_pct", 0.0) or 0.0
            )
        unstuck_patch = side_patch.get("unstuck", {}) or {}
        for offset, (patch_key, bot_key) in enumerate(
            (
                ("enabled", "unstuck_enabled"),
                ("ema_gating_enabled", "unstuck_ema_gating_enabled"),
                ("close_pct", "unstuck_close_pct"),
                ("ema_dist", "unstuck_ema_dist"),
                ("loss_allowance_pct", "unstuck_loss_allowance_pct"),
                ("threshold", "unstuck_threshold"),
            ),
            start=EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
        ):
            if patch_key in unstuck_patch:
                matrix[coin_index, offset] = float(effective_bot[bot_key])
        _pack_multicoin_hsl_overrides(
            matrix,
            row=coin_index,
            start_column=EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
            side_patch=side_patch,
            effective_bot=effective_bot,
        )
        if bool(effective_bot.get("is_forced_active", False)):
            matrix[
                coin_index, EMA_ANCHOR_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN
            ] = 1.0
    contract = {
        "exchange": exchange,
        "coins": coins,
        "side": side,
        "exact_overrides": exact_overrides,
        "values": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ],
    }
    return matrix, contract


def _build_multicoin_tm_coin_overrides(
    *,
    config: dict,
    mss: dict,
    exchange: str,
    coins: list[str],
    payload,
    side: str,
    resolve_override=None,
) -> tuple[np.ndarray, dict]:
    """Pack exact-last static coin overrides for the Metal TM proxy."""

    if resolve_override is None:
        from backtest import _get_backtest_coin_override

        resolve_override = _get_backtest_coin_override

    matrix = np.full(
        (len(coins), TRAILING_MARTINGALE_COIN_OVERRIDE_COLS),
        np.nan,
        dtype=np.float32,
    )
    exact_overrides = []
    missing = object()
    for coin_index, coin in enumerate(coins):
        patch = resolve_override(config, mss, exchange, coin) or {}
        exact_overrides.append(copy.deepcopy(patch))
        side_patch = patch.get("bot", {}).get(side, {})
        strategy_patch = (
            side_patch.get("strategy", {}).get("trailing_martingale", {}) or {}
        )
        effective_strategy = flatten_trailing_martingale_params(
            payload.strategy_params_list[coin_index][side],
            payload.bot_params_list[coin_index][side],
        )
        effective_bot = payload.bot_params_list[coin_index][side]
        for column, (key, path) in enumerate(
            TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
        ):
            value = strategy_patch
            for part in path:
                value = (
                    value.get(part, missing) if isinstance(value, dict) else missing
                )
                if value is missing:
                    break
            if value is not missing:
                effective_value = float(effective_strategy[key])
                matrix[coin_index, column] = (
                    encode_tm_retracement_base_pct(effective_value)
                    if key
                    in {
                        "entry_retracement_base_pct",
                        "close_retracement_base_pct",
                    }
                    else effective_value
                )
        entry_patch = strategy_patch.get("entry", {}) or {}
        if "ema_gate_mode" in entry_patch:
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN,
            ] = float(effective_strategy["gate_initial"])
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN,
            ] = float(effective_strategy["gate_reentry"])
        risk_patch = side_patch.get("risk", {}) or {}
        if "entry_cooldown_minutes" in risk_patch:
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
            ] = float(
                effective_bot.get("risk_entry_cooldown_minutes", 0.0) or 0.0
            )
        if not bool(effective_bot.get("entry_eligible", True)) or (
            "wallet_exposure_limit" in side_patch
        ):
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
            ] = float(
                effective_bot["wallet_exposure_limit"]
            )
        if "we_excess_allowance_pct" in risk_patch:
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN,
            ] = float(
                effective_bot.get("risk_we_excess_allowance_pct", 0.0) or 0.0
            )
        if "position_exposure_enforcer_enabled" in risk_patch:
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_ENABLED_COLUMN,
            ] = float(
                bool(effective_bot.get("risk_wel_enforcer_enabled", False))
            )
        if "position_exposure_enforcer_threshold" in risk_patch:
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_THRESHOLD_COLUMN,
            ] = float(
                effective_bot.get("risk_wel_enforcer_threshold", 0.0) or 0.0
            )
        unstuck_patch = side_patch.get("unstuck", {}) or {}
        for offset, (patch_key, bot_key) in enumerate(
            (
                ("enabled", "unstuck_enabled"),
                ("ema_gating_enabled", "unstuck_ema_gating_enabled"),
                ("close_pct", "unstuck_close_pct"),
                ("ema_dist", "unstuck_ema_dist"),
                ("loss_allowance_pct", "unstuck_loss_allowance_pct"),
                ("threshold", "unstuck_threshold"),
            ),
            start=TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
        ):
            if patch_key in unstuck_patch:
                matrix[coin_index, offset] = float(effective_bot[bot_key])
        _pack_multicoin_hsl_overrides(
            matrix,
            row=coin_index,
            start_column=TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
            side_patch=side_patch,
            effective_bot=effective_bot,
        )
        if bool(effective_bot.get("is_forced_active", False)):
            matrix[
                coin_index,
                TRAILING_MARTINGALE_COIN_OVERRIDE_FORCED_ACTIVE_COLUMN,
            ] = 1.0
    contract = {
        "exchange": exchange,
        "coins": coins,
        "side": side,
        "exact_overrides": exact_overrides,
        "values": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ],
    }
    return matrix, contract


def _build_single_coin_override_params(
    *,
    config: dict,
    mss: dict,
    exchange: str,
    coin: str,
    payload,
    side: str,
    strategy_kind: str,
) -> tuple[dict[str, float], dict]:
    """Map the shared exact-last override ABI onto one directional row."""

    if strategy_kind == "ema_anchor":
        matrix, contract = _build_multicoin_ema_coin_overrides(
            config=config,
            mss=mss,
            exchange=exchange,
            coins=[coin],
            payload=payload,
            side=side,
        )
        columns = {
            key: column
            for column, key in enumerate(EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS)
        }
        columns.update(
            {
                "entry_cooldown_minutes": EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
                "wallet_exposure_limit": (
                    EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
                ),
                "we_excess_allowance_pct": (
                    EMA_ANCHOR_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN
                ),
            }
        )
        unstuck_start = EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN
        hsl_start = EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN
    elif strategy_kind == "trailing_martingale":
        matrix, contract = _build_multicoin_tm_coin_overrides(
            config=config,
            mss=mss,
            exchange=exchange,
            coins=[coin],
            payload=payload,
            side=side,
        )
        columns = {
            key: column
            for column, (key, _path) in enumerate(
                TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
            )
        }
        columns.update(
            {
                "entry_cooldown_minutes": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN
                ),
                "wallet_exposure_limit": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
                ),
                "we_excess_allowance_pct": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_ALLOWANCE_PCT_COLUMN
                ),
                "wel_enforcer_enabled": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_ENABLED_COLUMN
                ),
                "wel_enforcer_threshold": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_WEL_ENFORCER_THRESHOLD_COLUMN
                ),
                "gate_initial": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_INITIAL_COLUMN
                ),
                "gate_reentry": (
                    TRAILING_MARTINGALE_COIN_OVERRIDE_GATE_REENTRY_COLUMN
                ),
            }
        )
        unstuck_start = TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN
        hsl_start = TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN
    else:
        raise ValueError(f"unsupported single-coin strategy {strategy_kind!r}")

    columns.update(
        {key: unstuck_start + offset for offset, key in enumerate(UNSTUCK_PARAM_KEYS)}
    )
    columns.update(
        {
            key: hsl_start + offset
            for offset, (key, _path) in enumerate(HSL_COIN_OVERRIDE_PATHS)
            if key != "hsl_panic_market"
        }
    )
    row = matrix[0]
    return (
        {
            key: float(row[column])
            for key, column in columns.items()
            if np.isfinite(row[column])
        },
        contract,
    )


class MpsMulticoinProxy:
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
    ):
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "GPU optimization requires the optional 'gpu-mps' dependencies; "
                "install Passivbot with `pip install -e '.[full,gpu-mps]'`"
            ) from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "GPU optimization requested but Apple MPS is unavailable in this process"
            )

        from optimization.gpu.metrics import (
            BTC_INTRADAY_RISK_METRICS,
            ENTRY_INTERVAL_METRICS,
            EQUITY_BALANCE_DIFF_METRICS,
            compute_objectives,
            validate_gpu_metric_names,
        )
        self.needed_metrics = set(validate_gpu_metric_names(needed_metrics))

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
        self.btc_risk_enabled = bool(
            self.needed_metrics & BTC_INTRADAY_RISK_METRICS
        )
        self.equity_balance_diff_enabled = bool(
            self.needed_metrics & EQUITY_BALANCE_DIFF_METRICS
        )
        btc_values = np.ascontiguousarray(
            np.asarray(btc, dtype=np.float64).reshape(-1)
        )
        self.batch_size = max(1, int(batch_size))
        self.max_dispatch_candidate_bars = int(max_dispatch_candidate_bars)
        self.interrupt_check = interrupt_check or (lambda: None)
        self.profile_enabled = os.environ.get(
            "PASSIVBOT_GPU_PROFILE", ""
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.last_profile: dict = {}

        values = np.asarray(hlcvs)
        if values.ndim != 3:
            raise ValueError(
                "expected multicoin HLCVs with three dimensions, "
                f"got {values.shape}"
            )
        coin_count = int(values.shape[1])
        if not (2 <= coin_count <= MPS_MULTICOIN_MAX_COINS):
            raise ValueError(
                f"MPS multicoin proxy supports 2..{MPS_MULTICOIN_MAX_COINS} coins; "
                f"got {coin_count}"
            )
        self.strategy_kind = (
            str(config.get("live", {}).get("strategy_kind", "")).strip().lower()
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
        self.param_keys = (
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
            if self.strategy_kind == "trailing_martingale"
            else EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        )
        enabled_sides = [
            side for side in ("long", "short") if gpu_side_enabled(config, side)
        ]
        if len(enabled_sides) not in (1, 2):
            raise ValueError(
                "MPS multicoin proxy requires one or two enabled sides"
            )
        self.sides = enabled_sides
        self.dispatch_batch_size = _mps_dispatch_batch_size(
            self.batch_size,
            n_bars=len(values),
            n_coins=coin_count,
            n_sides=len(enabled_sides),
            max_candidate_bars=self.max_dispatch_candidate_bars,
        )
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
        _require_multicoin_metric_topology(
            self.sides,
            self.needed_metrics,
            shared_account_controller=self.shared_account_fused,
        )
        payload = build_backtest_payload(
            np.ascontiguousarray(values),
            mss,
            copy.deepcopy(config),
            exchange,
            btc_values,
            timestamps,
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
        candle_interval_minutes = _single_coin_candle_interval_minutes(
            backtest_params
        )
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
                "live.forager_score_hysteresis_pct"
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
            "filter_volume_drop_pct",
            "forager_score_weights",
            "n_positions",
            "risk_twel_entry_gate_enabled",
            "risk_twel_enforcer_enabled",
            "risk_twel_enforcer_policy",
            "risk_twel_enforcer_threshold",
        )
        signal_mode = (
            backtest_params.get("equity_hard_stop_loss", {})
            .get("signal_mode", "unified")
        )
        hsl_enabled_sides = [
            side
            for side in self.sides
            if any(
                bool(item[side].get("hsl_enabled"))
                for item in payload.bot_params_list
            )
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
                    for item in payload.bot_params_list
                ],
                first_valid_indices=backtest_params["first_valid_indices"],
                last_valid_indices=backtest_params["last_valid_indices"],
            )
        self.base_total_wallet_exposure_limits = {
            side: float(
                payload.bot_params_list[0][side]["total_wallet_exposure_limit"]
            )
            for side in ("long", "short")
        }
        self.base_n_positions = {
            side: float(payload.bot_params_list[0][side]["n_positions"])
            for side in ("long", "short")
        }
        self.base_params = {}
        for side in self.sides:
            first_bot = payload.bot_params_list[0][side]
            first_strategy = dict(payload.strategy_params_list[0][side])
            if self.strategy_kind == "trailing_martingale":
                first_strategy = flatten_trailing_martingale_params(
                    first_strategy, first_bot
                )
            weights = first_bot.get("forager_score_weights", {}) or {}
            first_strategy.update(
                {
                    "entry_cooldown_minutes": float(
                        first_bot.get("risk_entry_cooldown_minutes", 0.0) or 0.0
                    ),
                    "total_wallet_exposure_limit": float(
                        first_bot["total_wallet_exposure_limit"]
                    ),
                    "forager_volume_ema_span_1m": float(
                        first_bot.get("filter_volume_ema_span_1m", 0.0) or 0.0
                    ),
                    "forager_volatility_ema_span_1m": float(
                        first_bot.get("filter_volatility_ema_span_1m", 0.0) or 0.0
                    ),
                    "forager_volume_drop_pct": float(
                        first_bot.get("filter_volume_drop_pct", 0.0) or 0.0
                    ),
                    "forager_score_weights_volume": float(
                        weights.get("volume", 0.0)
                    ),
                    "forager_score_weights_ema_readiness": float(
                        weights.get("ema_readiness", 0.0)
                    ),
                    "forager_score_weights_volatility": float(
                        weights.get("volatility", 0.0)
                    ),
                    "n_positions": float(first_bot["n_positions"]),
                }
            )
            first_strategy.update(
                _single_coin_exposure_params(
                    config["bot"][side].get("risk", {}), side=side
                )
            )
            if self.strategy_kind == "trailing_martingale":
                first_strategy.update(
                    _position_exposure_enforcer_params(
                        config["bot"][side].get("risk", {}), side=side
                    )
                )
            if self.strategy_kind in {"ema_anchor", "trailing_martingale"}:
                first_strategy.update(
                    _total_exposure_enforcer_params(
                        config["bot"][side].get("risk", {}), side=side
                    )
                )
            base_bot = flatten_shared_bot_side(config["bot"][side])
            first_strategy.update(_unstuck_params(base_bot))
            first_strategy.update(_hsl_params(base_bot, signal_mode=signal_mode))
            missing = [
                key for key in self.param_keys if key not in first_strategy
            ]
            if missing:
                raise ValueError(
                    f"MPS multicoin {self.strategy_kind} {side} payload is "
                    f"missing parameters: {missing}"
                )
            self.base_params[side] = first_strategy

            for coin in range(1, coin_count):
                bot = payload.bot_params_list[coin][side]
                if any(
                    bot.get(key) != first_bot.get(key)
                    for key in comparable_bot_keys
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

        coins = list(backtest_params.get("coins") or [])
        if len(coins) != coin_count:
            raise ValueError(
                "MPS multicoin payload coin identity disagrees with prepared data: "
                f"coins={coins}, prepared={coin_count}"
            )
        per_side_coin_overrides = {}
        per_side_override_contracts = {}
        for side in self.sides:
            if self.strategy_kind == "ema_anchor":
                overrides, contract = _build_multicoin_ema_coin_overrides(
                    config=config,
                    mss=mss,
                    exchange=exchange,
                    coins=coins,
                    payload=payload,
                    side=side,
                )
            else:
                overrides, contract = _build_multicoin_tm_coin_overrides(
                    config=config,
                    mss=mss,
                    exchange=exchange,
                    coins=coins,
                    payload=payload,
                    side=side,
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
                self.coin_override_contract["hsl_signal_mode"] = str(
                    signal_mode
                ).strip().lower()
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
            )
            for coin in range(coin_count)
        ]
        self.run = replace(
            runs[0],
            first_valid_idx=min(run.first_valid_idx for run in runs),
            last_valid_idx=max(run.last_valid_idx for run in runs),
            trade_start_idx=min(run.trade_start_idx for run in runs),
        )
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
        self.data = build_mps_multicoin_data(
            values,
            timestamps,
            runs=runs,
            markets=markets,
            include_hourly_ranges=True,
        )
        self.metrics_data = {
            "ts0": self.data["ts0"],
            "n": self.data["n"],
            "strategy_kind": self.strategy_kind,
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
        common_runner_kwargs = {
            "forager_score_hysteresis_pct": self.forager_score_hysteresis_pct,
            "max_realized_loss_pct": float(
                backtest_params.get("max_realized_loss_pct", 1.0)
            ),
            "collect_coin_fill_counts": bool(
                self.needed_metrics
                & {"fills_active_symbols_count", "fills_top_symbol_share"}
            ),
            "filter_by_min_effective_cost": bool(
                backtest_params.get("filter_by_min_effective_cost", False)
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
            "hsl_ema_tail_enabled": bool(
                self.needed_metrics & _HSL_EMA_TAIL_METRICS
            ),
            "hsl_raw_drawdown_enabled": bool(
                self.needed_metrics
                & (_HSL_RAW_DRAWDOWN_METRICS | _HSL_RAW_TAIL_METRICS)
            ),
            "hsl_raw_tail_enabled": bool(
                self.needed_metrics & _HSL_RAW_TAIL_METRICS
            ),
            "recovery_distribution_enabled": bool(
                self.needed_metrics
                & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS
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
        if self.shared_account_fused:
            fused_runner_cls = (
                MpsTrailingMartingaleMulticoinFusedRunner
                if self.strategy_kind == "trailing_martingale"
                else MpsEmaAnchorMulticoinFusedRunner
            )
            self.fused_runner = fused_runner_cls(
                self.run,
                self.data,
                long_coin_overrides=per_side_coin_overrides["long"],
                short_coin_overrides=per_side_coin_overrides["short"],
                hsl_panic_market_long=str(
                    flatten_shared_bot_side(config["bot"]["long"]).get(
                        "hsl_panic_close_order_type", "limit"
                    )
                ).strip().lower()
                == "market",
                hsl_panic_market_short=str(
                    flatten_shared_bot_side(config["bot"]["short"]).get(
                        "hsl_panic_close_order_type", "limit"
                    )
                ).strip().lower()
                == "market",
                hedge_mode=bool(backtest_params["hedge_mode"]),
                **common_runner_kwargs,
            )
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
                        flatten_shared_bot_side(config["bot"][side]).get(
                            "hsl_panic_close_order_type", "limit"
                        )
                    ).strip().lower()
                    == "market",
                    **common_runner_kwargs,
                }
                runner_kwargs["coin_overrides"] = per_side_coin_overrides[side]
                self.runners[side] = runner_cls(
                    self.run,
                    self.data,
                    **runner_kwargs,
                )

    def _parameter_matrix(
        self, candidates: list[dict], side: str | None = None
    ) -> np.ndarray:
        if side is None:
            if len(self.sides) != 1:
                raise ValueError("side is required for dual-side multicoin parameters")
            side = self.sides[0]
        param_keys = getattr(self, "param_keys", EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        rows = []
        for candidate in candidates:
            merged = dict(self.base_params[side])
            merged.update(
                {
                    key.removeprefix(f"{side}_"): value
                    for key, value in candidate.items()
                    if key.startswith(f"{side}_")
                }
            )
            rows.append([float(merged[key]) for key in param_keys])
        return np.asarray(rows, dtype=np.float64)

    def evaluate(self, candidates: list[dict]) -> list[dict]:
        results: list[dict] = []
        torch = self._torch
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
                    (
                        int(getattr(runner, "n_coins", 1))
                        for runner in profile_runners
                    ),
                    default=1,
                ),
                side_count=len(self.sides),
            )
            if self.profile_enabled
            else None
        )
        self.last_profile = {}
        dispatch_batch_size = getattr(
            self, "dispatch_batch_size", self.batch_size
        )
        interrupt_check = getattr(self, "interrupt_check", lambda: None)
        progress = _new_gpu_dispatch_progress(
            len(candidates), dispatch_batch_size
        )
        for start in range(0, len(candidates), dispatch_batch_size):
            chunk_profile_started = (
                time.perf_counter() if profile is not None else 0.0
            )
            interrupt_check()
            chunk = candidates[start : start + dispatch_batch_size]
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
                stage_started = (
                    time.perf_counter() if self.profile_enabled else 0.0
                )
                recovery_distribution = _mps_strategy_eq_recovery_distribution(
                    raw_output, self.needed_metrics
                )
                if profile is not None:
                    torch.mps.synchronize()
                    profile["timings_seconds"]["metric_reduction"] += (
                        time.perf_counter() - stage_started
                    )
                    stage_started = time.perf_counter()
                output = {
                    key: value.cpu()
                    for key, value in raw_output.items()
                    if key in CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS
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
                stage_started = (
                    time.perf_counter() if self.profile_enabled else 0.0
                )
                recovery_distribution = (
                    _mps_strategy_eq_recovery_distribution(
                        raw_side_outputs[self.sides[0]], self.needed_metrics
                    )
                    if len(self.sides) == 1
                    else None
                )
                if profile is not None:
                    torch.mps.synchronize()
                    profile["timings_seconds"]["metric_reduction"] += (
                        time.perf_counter() - stage_started
                    )
                    stage_started = time.perf_counter()
                side_outputs = {
                    side: {
                        key: value.cpu()
                        for key, value in raw_side_outputs[side].items()
                        if key in CORE_OUTPUT_KEYS | DIRECTIONAL_HSL_OUTPUT_KEYS
                    }
                    for side in self.sides
                }
                if recovery_distribution is not None:
                    recovery_distribution = recovery_distribution.cpu()
                if profile is not None:
                    profile["timings_seconds"]["device_to_host"] += (
                        time.perf_counter() - stage_started
                    )
            stage_started = (
                time.perf_counter() if self.profile_enabled else 0.0
            )
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
            elif fused_runner is None:
                output = _combine_hedged_multicoin_outputs(
                    side_outputs["long"],
                    side_outputs["short"],
                    self.run.starting_balance,
                    self.run.liquidation_threshold,
                    self.runners["long"].start_minute_of_day,
                    self.run.interval_ms,
                )
                if any(
                    name.startswith("hard_stop_") for name in self.needed_metrics
                ) and _refresh_hedged_multicoin_hsl_at_portfolio_cutoff(
                    side_outputs=side_outputs,
                    runners=self.runners,
                    parameter_matrices=parameter_matrices,
                    combined_output=output,
                    start_minute_of_day=self.runners[
                        "long"
                    ].start_minute_of_day,
                    interrupt_check=interrupt_check,
                    profile=self.profile_enabled,
                    runner_profile_callback=(
                        lambda runner, **kwargs: _add_gpu_runner_profile(
                            profile, runner, **kwargs
                        )
                        if profile is not None
                        else None
                    ),
                    profile_timings=(
                        profile["timings_seconds"] if profile is not None else None
                    ),
                ):
                    output = _combine_hedged_multicoin_outputs(
                        side_outputs["long"],
                        side_outputs["short"],
                        self.run.starting_balance,
                        self.run.liquidation_threshold,
                        self.runners["long"].start_minute_of_day,
                        self.run.interval_ms,
                    )
                output["entry_initial_balance_pct_long"] = side_outputs[
                    "long"
                ]["entry_initial_balance_pct"]
                output["entry_initial_balance_pct_short"] = side_outputs[
                    "short"
                ]["entry_initial_balance_pct"]
            if recovery_distribution is not None:
                output["strategy_eq_recovery_distribution"] = recovery_distribution
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
            objectives = self._compute_objectives(
                output, self.run, self.metrics_data, needed=self.needed_metrics
            )
            if profile is not None:
                profile["timings_seconds"]["metric_reduction"] += (
                    _gpu_profile_unattributed_seconds(
                        profile["timings_seconds"],
                        time.perf_counter() - stage_started,
                        device_to_host_before=stage_device_to_host_before,
                        runner_seconds_before=stage_runner_seconds_before,
                    )
                )
                stage_started = time.perf_counter()
            arrays = {
                name: value.detach().cpu().numpy()
                for name, value in objectives.items()
            }
            results.extend(
                {name: float(values[index]) for name, values in arrays.items()}
                for index in range(len(chunk))
            )
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
            self.last_profile = _finish_gpu_proxy_profile(
                profile, profile_started
            )
        return results


# Compatibility alias retained for callers and tests from the EMA-only slices.
MpsMulticoinEmaProxy = MpsMulticoinProxy
