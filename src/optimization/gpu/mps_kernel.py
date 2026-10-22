from __future__ import annotations

from config.policy import validate_minimum_cost_filters
from config.activation import validate_side_activation
from functools import lru_cache
import time
import logging

import numpy as np
import torch

from optimization.gpu.ema_tail import (
    drawdown_ema_tail_from_samples, drawdown_ema_tail_history_bytes,
)
from optimization.gpu.ema_bootstrap import (
    BOOTSTRAP_COUNT_COLS, effective_end_steps, pack_bootstrap_counts, validate_hourly_bootstrap_inputs,
)
from optimization.gpu.entry_intervals import (
    entry_interval_history_bytes, entry_intervals_from_counts,
)

from optimization.gpu.specialization import unstuck_ema_required, unstuck_required
from optimization.gpu.decimal import (
    _TM_DECIMAL_KEYS,
    _TM_DECIMAL_WORDS,
    _pack_tm_parameter_matrix,
    _tm_decimal_metadata,
    _tm_modifier_masks,
)
from optimization.gpu.replay_progress import TemporalReplayProgress
from optimization.gpu.autotune import record_replay_chunk, ReplayDurationController
from optimization.gpu.runtime import (
    gpu_device,
    compile_shader,
    synchronize,
    wait_for_cuda_stream,
)

from optimization.gpu.model import (
    MARKET_EXECUTION_TICKS_ABI,
    validate_market_order_slippage_pct,
    ADAPTIVE_PARAM_KEYS,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    GAP_BINS,
    MPS_MULTICOIN_MAX_COINS,
    MPS_TM_MULTICOIN_CHUNK_BARS,
    MPS_TM_MULTICOIN_CHUNK_CANDIDATE_STEPS,
    CUDA_FACTUAL_HSL_REPLAY_CHUNK_BARS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
)

MPS_DAILY_COLS = 8
MPS_MULTICOIN_DAILY_COLS = 9
MPS_MULTICOIN_BASE_SCALAR_COLS = 62
MPS_MULTICOIN_EMA_TAIL_SCALAR_COLS = 64
MPS_MULTICOIN_RAW_DRAWDOWN_SCALAR_COLS = 66
MPS_MULTICOIN_SCALAR_COLS = 68
MPS_MULTICOIN_FUSED_BASE_SCALAR_COLS = 67
MPS_MULTICOIN_FUSED_EMA_TAIL_SCALAR_COLS = 69
MPS_MULTICOIN_FUSED_RAW_DRAWDOWN_SCALAR_COLS = 71
MPS_MULTICOIN_FUSED_SCALAR_COLS = 73
# A 30-day coin-HSL lookback can legitimately contain slightly more than
# 2,048 completed round trips for high-cadence single-coin candidates. Metal
# coalesces every realized-PnL component from one candle into one ring event,
# so ladder fill multiplicity does not consume extra slots. Keep this bounded,
# but leave enough headroom for dense valid event-candle windows.
MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS = 7
MPS_EQUITY_BALANCE_DIFF_COLS = 12
MPS_ENTRY_INTERVAL_STAT_COLS = 2
MPS_ENTRY_INTERVAL_COUNT_COLS = 129

_HSL_EMA_TAIL_DEFINE = "#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 1\n"
_HSL_RAW_DRAWDOWN_DEFINE = "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n"
_HSL_RAW_TAIL_DEFINE = "#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 1\n"
_HSL_DIAGNOSTICS_DISABLE_DEFINE = "#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 0\n"
_HSL_DISABLED_DEFINE = "#define PASSIVBOT_HSL_DISABLED 1\n"
_RECOVERY_DISTRIBUTION_DEFINE = (
    "#define PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED 1\n"
)
_FIXED_WEL_DENOMINATOR_DEFINE = "#define PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY 0\n"
_BTC_RISK_DEFINE = "#define PASSIVBOT_BTC_RISK_ENABLED 1\n"
_EQUITY_BALANCE_DIFF_DEFINE = "#define PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED 1\n"
_ENTRY_INTERVAL_DEFINE = "#define PASSIVBOT_ENTRY_INTERVAL_ENABLED 1\n"
_TM_LOSS_GATE_DISABLED_DEFINE = "#define PASSIVBOT_TM_LOSS_GATE_DISABLED 1\n"


def _hsl_layout(capacity: int, fact_capacity: int = 0, *, factual_only: bool = False) -> tuple[int, int]:
    blocks = (capacity + 63) // 64
    tree_size = 1 << (blocks - 1).bit_length()
    if type(fact_capacity) is not int or fact_capacity < 0:
        raise ValueError("Invalid GPU HSL factual capacity")
    if factual_only:
        nodes = 2 + fact_capacity + (fact_capacity + 1) // 2 if fact_capacity else 0
        return tree_size, nodes
    storage_nodes = 2 * tree_size + (capacity + 3) // 4
    if fact_capacity:
        storage_nodes += 2 + fact_capacity + (fact_capacity + 1) // 2
    return tree_size, storage_nodes


def _hsl_native_cache_bytes(capacity: int, pair_capacity: int, fact_capacity: int) -> int:
    """Exact one-owner scratch layout, independently of the number of controllers."""
    if any(type(value) is not int or value < 0 for value in (capacity, pair_capacity, fact_capacity)):
        raise ValueError("Invalid native HSL cache dimensions")
    if not capacity or not pair_capacity or not fact_capacity:
        return 0
    if capacity % 64:
        raise ValueError("Native HSL cache capacity must contain complete blocks")
    align = lambda size: (size + 15) // 16 * 16
    return sum(align(size) for size in (
        96, pair_capacity * 128, capacity * 16, capacity * 8, capacity * 4,
        capacity * 28, (capacity // 64) * 32, fact_capacity * 16,
    ))


def _with_hsl(source: str, capacity: int, fact_capacity: int = 0) -> str:
    if fact_capacity and not capacity:
        raise ValueError("GPU HSL facts require an HSL window")
    if not capacity:
        return source
    if not 1 <= capacity <= 90 * 1440 + 2:
        raise ValueError("Invalid GPU HSL window capacity")
    tree_size, _ = _hsl_layout(capacity, fact_capacity)
    return (
        "#define PASSIVBOT_HSL_FACTS_ENABLED 0\n"
        f"#define PASSIVBOT_HSL 1\n"
        f"#define PASSIVBOT_HSL_CAPACITY {capacity}\n"
        f"#define PASSIVBOT_HSL_TREE_SIZE {tree_size}\n" + source
    )


def _with_hsl_ema_tail(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "#ifndef PASSIVBOT_HSL_EMA_TAIL_ENABLED" not in source:
        raise RuntimeError("MPS source is missing the HSL EMA-tail feature guard")
    return _HSL_EMA_TAIL_DEFINE + source


def _raw_drawdown_tail_capacity(n_days: int) -> int:
    """Bound the worst 1% by the prepared UTC horizon, sharing power-of-two variants."""
    if type(n_days) is not int or n_days < 1:
        raise ValueError("raw drawdown tail requires a positive prepared day count")
    needed = max(n_days // 100, 1)
    return 1 << (needed - 1).bit_length()


def _with_hsl_features(
    source: str,
    *,
    ema_tail_enabled: bool,
    raw_drawdown_enabled: bool,
    raw_tail_enabled: bool,
    diagnostics_enabled: bool = True,
    raw_tail_capacity: int = 1,
) -> str:
    if raw_tail_enabled and not raw_drawdown_enabled:
        raise ValueError("HSL raw-tail metrics require raw-drawdown metrics")
    if not diagnostics_enabled and (
        ema_tail_enabled or raw_drawdown_enabled or raw_tail_enabled
    ):
        raise ValueError("HSL diagnostic feature outputs require diagnostics")
    if not diagnostics_enabled:
        if "#ifndef PASSIVBOT_HSL_DIAGNOSTICS_ENABLED" not in source:
            raise RuntimeError(
                "MPS source is missing the HSL diagnostics feature guard"
            )
        source = _HSL_DIAGNOSTICS_DISABLE_DEFINE + source
    source = _with_hsl_ema_tail(source, ema_tail_enabled)
    if raw_drawdown_enabled:
        if "#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED" not in source:
            raise RuntimeError(
                "MPS source is missing the HSL raw-drawdown feature guard"
            )
        source = _HSL_RAW_DRAWDOWN_DEFINE + source
    if raw_tail_enabled:
        if type(raw_tail_capacity) is not int or raw_tail_capacity < 1:
            raise ValueError("HSL raw-tail capacity must be a positive integer")
        if "#ifndef PASSIVBOT_HSL_RAW_TAIL_ENABLED" not in source:
            raise RuntimeError("MPS source is missing the HSL raw-tail feature guard")
        source = (f"#define PASSIVBOT_HSL_RAW_TAIL_CAPACITY {raw_tail_capacity}\n"
                  + _HSL_RAW_TAIL_DEFINE + source)
    return source


def _with_hsl_disabled(source: str, disabled: bool) -> str:
    """Compile away per-coin HSL state for a proven disabled dispatch."""

    if not disabled:
        return source
    if "#if PASSIVBOT_HSL_DISABLED" not in source:
        raise RuntimeError("MPS source is missing the disabled-HSL feature guard")
    return _HSL_DISABLED_DEFINE + source


def _with_recovery_distribution(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "#ifdef PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED" not in source:
        raise RuntimeError(
            "MPS source is missing the strategy-equity recovery-distribution feature guard"
        )
    return _RECOVERY_DISTRIBUTION_DEFINE + source


def _with_dynamic_wel_by_tradability(source: str, enabled: bool) -> str:
    if enabled:
        return source
    if "#ifndef PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY" not in source:
        raise RuntimeError(
            "MPS multicoin source is missing the dynamic-WEL feature guard"
        )
    return _FIXED_WEL_DENOMINATOR_DEFINE + source


def _with_btc_risk(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "struct BtcRiskState" not in source:
        raise RuntimeError("MPS source is missing the shared BTC-risk contract")
    return _BTC_RISK_DEFINE + source


def _with_equity_balance_diff(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "struct EquityBalanceDiffState" not in source:
        raise RuntimeError(
            "MPS source is missing the shared equity-balance-diff contract"
        )
    return _EQUITY_BALANCE_DIFF_DEFINE + source


def _with_entry_interval(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "inline void record_initial_entry_interval(" not in source:
        raise RuntimeError("MPS source is missing the shared entry-interval contract")
    return _ENTRY_INTERVAL_DEFINE + source


def _with_tm_loss_gate_disabled(source: str, disabled: bool) -> str:
    if not disabled:
        return source
    if "#ifndef PASSIVBOT_TM_LOSS_GATE_DISABLED" not in source:
        raise RuntimeError("GPU source is missing the TM loss-gate specialization contract")
    return _TM_LOSS_GATE_DISABLED_DEFINE + source


def _encode_max_realized_loss_pct(value: float) -> float:
    """Match the shared risk graph's nearest binary32 fraction cast.

    Original float64 enablement is transported separately: a fraction just below
    one may round to one while its configured gate remains enabled.
    """

    return 1.0 if value >= 1.0 else float(np.float32(value))


def _btc_risk_price_tensor(btc_prices, *, expected_count: int):
    if btc_prices is None:
        return None
    values = np.ascontiguousarray(np.asarray(btc_prices, dtype=np.float32).reshape(-1))
    if len(values) != int(expected_count):
        raise ValueError(
            "MPS BTC-risk prices must match the prepared candle count: "
            f"btc={len(values)}, candles={expected_count}"
        )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "MPS BTC-risk prices must remain finite and positive after float32 packing"
        )
    return torch.as_tensor(values, dtype=torch.float32, device=gpu_device())







def _validate_unstuck_ema_spans(
    values: np.ndarray, *, allow_unset: bool = False
) -> None:
    if allow_unset:
        values = values[~np.isnan(values)]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        packed = values.astype(np.float32)
    if np.any(~np.isfinite(packed)) or np.any(packed <= 0.0):
        raise ValueError(
            "MPS unstuck EMA spans must remain positive and finite in float32 candle periods"
        )


def _scale_directional_minute_parameters(
    params: np.ndarray,
    keys: tuple[str, ...],
    *,
    sides: int,
    interval_minutes: float,
    ranking_coin_counts: tuple[int, ...] | None = None,
    cooldown_coin_overrides: tuple[np.ndarray, ...] | None = None,
    dynamic_wel_by_tradability: bool = True,
    side_activation: tuple[bool, ...] | None = None,
) -> np.ndarray:
    """Convert minute-denominated directional inputs to candle periods.

    Exact Rust divides strategy EMA spans by the configured candle interval
    before calculating their alphas. It separately compounds HSL's one-minute
    EMA decay over the elapsed minutes, so HSL spans are converted to the
    equivalent per-candle alpha. Entry and HSL cooldowns remain expressed in
    elapsed minutes and are converted to candle counts. Hour-denominated
    volatility spans remain unchanged.
    """

    interval_minutes = float(interval_minutes)
    if not np.isfinite(interval_minutes) or interval_minutes < 1.0:
        raise ValueError("MPS candle interval must be finite and at least one minute")
    scaled = np.array(params, dtype=np.float64, copy=True)
    minute_keys = {
        "unstuck_ema_span_0",
        "unstuck_ema_span_1",
        "ema_span_0",
        "ema_span_1",
        "entry_cooldown_minutes",
    }
    minute_keys.update(ADAPTIVE_PARAM_KEYS[:4])
    if "offset_volatility_ema_span_1m" in keys:
        minute_keys.add("offset_volatility_ema_span_1m")
    if "volatility_ema_span_1m" in keys:
        minute_keys.add("volatility_ema_span_1m")
    if "forager_volume_ema_span_1m" in keys:
        minute_keys.add("forager_volume_ema_span_1m")
    if "forager_volatility_ema_span_1m" in keys:
        minute_keys.add("forager_volatility_ema_span_1m")
    side_width = len(keys)
    if side_activation is not None and (len(side_activation) != sides
            or any(type(flag) is not bool for flag in side_activation)):
        raise TypeError("GPU scaling requires one activation boolean per side")
    for side_index in range(sides):
        offset = side_index * side_width
        span_col = offset + keys.index("unilateralness_ema_span_1m")
        adverse = scaled[:, offset + keys.index("entry_cooldown_adverse_weight")]
        score = scaled[:, offset + keys.index("forager_score_weights_unilateralness")]
        ceiling = scaled[:, offset + keys.index("entry_cooldown_max_duration_minutes")]
        floor = scaled[:, offset + keys.index("entry_cooldown_min_duration_minutes")]
        base = scaled[:, offset + keys.index("entry_cooldown_minutes")]
        exposure = scaled[:, offset + keys.index("entry_cooldown_exposure_weight")]
        policy = np.column_stack((floor, ceiling, exposure, adverse))
        with np.errstate(over="ignore"):
            finite_policy = np.isfinite(policy.astype(np.float32)).all()
        if (
            not finite_policy
            or np.any(floor < 0)
            or np.any(exposure < 0)
            or np.any(adverse < 0)
            or np.any((ceiling < 0) & (ceiling != -1))
            or np.any((ceiling >= 0) & (ceiling < floor))
            or np.any(((exposure > 0) | (adverse > 0)) & (ceiling < 0))
        ):
            raise ValueError(
                "GPU adaptive cooldown requires finite nonnegative weights/floor and a valid bounded ceiling"
            )
        constant = (ceiling >= 0.0) & (np.maximum(base, floor) >= ceiling)
        ranking = np.zeros(len(scaled), dtype=bool)
        if "n_positions" in keys:
            ranking = score > 0.0
            if ranking_coin_counts is not None:
                count = ranking_coin_counts[side_index]
                slots = np.rint(scaled[:, offset + keys.index("n_positions")])
                ranking &= (count > 1) & (dynamic_wel_by_tradability | (count > slots))
                # Match Rust's dormant scoring policy. Aggregated candles must
                # never update a one-minute indicator that cannot be consumed.
                scaled[~ranking, offset + keys.index("forager_score_weights_unilateralness")] = 0.0
        adverse_demand = (adverse > 0.0) & ~constant
        if cooldown_coin_overrides is not None:
            # Override rows are unscaled base/floor/ceiling/adverse/WEL pins.
            # Resolve against each candidate before deciding whether RMS is used.
            adverse_demand = np.zeros(len(scaled), dtype=bool)
            overrides = cooldown_coin_overrides[side_index]
            # Avoid per-coin work when all adverse weights are disabled.
            if np.any(adverse > 0.0) or np.any(overrides[:, 3] > 0.0):
                for pins in overrides:
                    effective_adverse = pins[3] if np.isfinite(pins[3]) else adverse
                    if pins[4] == 0.0 or not np.any(effective_adverse > 0.0):
                        continue
                    effective_base = pins[0] if np.isfinite(pins[0]) else base
                    effective_floor = pins[1] if np.isfinite(pins[1]) else floor
                    effective_ceiling = pins[2] if np.isfinite(pins[2]) else ceiling
                    effective_constant = (effective_ceiling >= 0.0) & (
                        np.maximum(effective_base, effective_floor) >= effective_ceiling
                    )
                    adverse_demand |= (effective_adverse > 0.0) & ~effective_constant
        rms = adverse_demand | ranking
        if side_activation is not None:
            rms &= side_activation[side_index]
        rms &= scaled[:, offset + keys.index("total_wallet_exposure_limit")] > 0.0
        if np.any(rms) and interval_minutes != 1.0:
            raise ValueError("GPU RMS directionality requires one-minute candles")
        spans = scaled[:, span_col]
        if np.any(rms & (~np.isfinite(spans) | (spans < 1.0) | (spans > 100000.0))):
            raise ValueError(
                "GPU unilateralness spans must be finite and between 1 and 100000"
            )
        scaled[:, offset + keys.index("unilateralness_window")] = np.ceil(20.0 * spans)
        for key in minute_keys:
            column = offset + keys.index(key)
            if key == "entry_cooldown_max_duration_minutes":
                finite_ceiling = scaled[:, column] >= 0.0
                scaled[finite_ceiling, column] /= interval_minutes
            else:
                scaled[:, column] /= interval_minutes
        for key in ("unstuck_ema_span_0", "unstuck_ema_span_1"):
            _validate_unstuck_ema_spans(scaled[:, offset + keys.index(key)])
        if interval_minutes != 1.0:
            hsl_span_column = offset + keys.index("hsl_drawdown_ratio_ema_span_minutes")
            hsl_spans = scaled[:, hsl_span_column]
            if np.any(~np.isfinite(hsl_spans)) or np.any(hsl_spans < 1.0):
                raise ValueError(
                    "MPS HSL EMA span must be finite and at least one minute"
                )
            alpha_1m = 2.0 / (hsl_spans + 1.0)
            decay_1m = 1.0 - alpha_1m
            alpha_per_candle = np.ones_like(decay_1m)
            positive_decay = decay_1m > 0.0
            alpha_per_candle[positive_decay] = -np.expm1(
                interval_minutes * np.log(decay_1m[positive_decay])
            )
            scaled[:, hsl_span_column] = 2.0 / alpha_per_candle - 1.0
    return scaled




def _scale_multicoin_coin_overrides(
    coin_overrides: np.ndarray,
    interval_minutes: float,
    *,
    expected_cols: int,
    label: str,
    minute_columns: set[int],
    hsl_start_column: int,
) -> np.ndarray:
    """Convert finite exact-last minute overrides to candle periods."""

    scaled = np.array(coin_overrides, dtype=np.float64, copy=True)
    if scaled.ndim != 2 or scaled.shape[1] != expected_cols:
        raise ValueError(
            f"expected multicoin {label} override matrix with "
            f"{expected_cols} columns, got {scaled.shape}"
        )
    interval_minutes = float(interval_minutes)
    if not np.isfinite(interval_minutes) or interval_minutes < 1.0:
        raise ValueError(
            "MPS candle interval must be finite and at least one minute"
        )
    adaptive_start = expected_cols - 4
    # RMS demand depends on the effective candidate plus these pins. Validate it
    # when packing the candidate batch, not from incomplete overrides alone.
    for column in minute_columns | {
        adaptive_start,
        adaptive_start + 1,
        adaptive_start + 2,
        adaptive_start + 3,
    }:
        finite = np.isfinite(scaled[:, column])
        if column == adaptive_start + 1:
            finite &= scaled[:, column] >= 0.0
        scaled[finite, column] /= interval_minutes
    if interval_minutes != 1.0:
        hsl_span_column = hsl_start_column + 2
        finite = np.isfinite(scaled[:, hsl_span_column])
        hsl_spans = scaled[finite, hsl_span_column]
        if np.any(hsl_spans < 1.0):
            raise ValueError("MPS HSL EMA span override must be at least one minute")
        alpha_1m = 2.0 / (hsl_spans + 1.0)
        decay_1m = 1.0 - alpha_1m
        alpha_per_candle = np.ones_like(decay_1m)
        positive_decay = decay_1m > 0.0
        alpha_per_candle[positive_decay] = -np.expm1(
            interval_minutes * np.log(decay_1m[positive_decay])
        )
        scaled[finite, hsl_span_column] = 2.0 / alpha_per_candle - 1.0
    _validate_unstuck_ema_spans(scaled[:, -6:-4], allow_unset=True)
    return np.ascontiguousarray(scaled, dtype=np.float32)


def _scale_ema_multicoin_coin_overrides(
    coin_overrides: np.ndarray, interval_minutes: float
) -> np.ndarray:
    """Convert finite exact-last EMA coin overrides to candle periods."""

    return _scale_multicoin_coin_overrides(
        coin_overrides,
        interval_minutes,
        expected_cols=EMA_ANCHOR_COIN_OVERRIDE_COLS,
        label="EMA",
        minute_columns={
            EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
            EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN + 1,
            EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS.index("ema_span_0"),
            EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS.index("ema_span_1"),
            EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS.index(
                "offset_volatility_ema_span_1m"
            ),
            EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
        },
        hsl_start_column=EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    )


def _scale_tm_multicoin_coin_overrides(
    coin_overrides: np.ndarray, interval_minutes: float
) -> np.ndarray:
    """Convert finite exact-last TM coin overrides to candle periods."""

    override_keys = tuple(key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS)
    return _scale_multicoin_coin_overrides(
        coin_overrides,
        interval_minutes,
        expected_cols=TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
        label="Trailing Martingale",
        minute_columns={
            TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN,
            TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN + 1,
            override_keys.index("ema_span_0"),
            override_keys.index("ema_span_1"),
            override_keys.index("volatility_ema_span_1m"),
            TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
        },
        hsl_start_column=TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    )


def _scalar_column_or_zero(scalars, index: int, *, reserved_columns: int = 2):
    # Feature-specialized shared replay reserves the portfolio tail and moments.
    if scalars.shape[1] - reserved_columns > index:
        return scalars[:, index]
    return torch.zeros_like(scalars[:, 0])


def _decode_btc_risk_outputs(daily, active_days, first_column: int) -> dict:
    if daily.shape[2] < first_column + 3:
        return {}
    return {
        "btc_day_end_eq": daily[:, :, first_column],
        "btc_day_min_eq": torch.where(
            active_days,
            daily[:, :, first_column + 1],
            torch.full_like(daily[:, :, first_column + 1], float("inf")),
        ),
        "btc_day_max_dd": daily[:, :, first_column + 2],
    }


def _decode_equity_balance_diff_outputs(values) -> dict:
    if values is None:
        return {}
    if values.ndim != 2 or values.shape[1] != MPS_EQUITY_BALANCE_DIFF_COLS:
        raise RuntimeError(
            "MPS equity-balance-diff output has an invalid shape: "
            f"{tuple(values.shape)}"
        )
    output = {}
    for suffix, offset in (("", 0), ("_btc", 6)):
        positive_count = values[:, offset + 2]
        negative_count = values[:, offset + 5]
        zeros = torch.zeros_like(positive_count)
        output[f"equity_balance_diff_pos_max{suffix}"] = values[:, offset]
        output[f"equity_balance_diff_pos_mean{suffix}"] = torch.where(
            positive_count > 0.0,
            values[:, offset + 1] / positive_count.clamp(min=1.0),
            zeros,
        )
        output[f"equity_balance_diff_neg_max{suffix}"] = values[:, offset + 3]
        output[f"equity_balance_diff_neg_mean{suffix}"] = torch.where(
            negative_count > 0.0,
            values[:, offset + 4] / negative_count.clamp(min=1.0),
            zeros,
        )
    return output


def _decode_entry_interval_outputs(stats, counts) -> dict:
    if stats is None and counts is None:
        return {}
    if stats is None or counts is None:
        raise RuntimeError("MPS entry-interval output is only partially present")
    if stats.ndim != 2 or stats.shape[1] != MPS_ENTRY_INTERVAL_STAT_COLS:
        raise RuntimeError(
            "MPS entry-interval stats have an invalid shape: " f"{tuple(stats.shape)}"
        )
    if counts.ndim != 2 or counts.shape[1] != MPS_ENTRY_INTERVAL_COUNT_COLS:
        raise RuntimeError(
            "MPS entry-interval counts have an invalid shape: " f"{tuple(counts.shape)}"
        )
    return {
        "entry_interval_sum_steps": stats[:, 0],
        "entry_interval_count": counts[:, 0],
        "entry_interval_max_steps": stats[:, 1],
        "entry_interval_hist": counts[:, 1:],
    }


def _lookup_replay_kernel(library, name, *, profile):
    """Metal creates its compute pipeline lazily during kernel lookup."""
    started = time.perf_counter() if profile else 0.0
    kernel = getattr(library, name)
    finished = time.perf_counter() if profile else 0.0
    return kernel, finished - started, finished


def _cached_library_with_miss(loader, *args):
    misses_before = loader.cache_info().misses
    library = loader(*args)
    return library, loader.cache_info().misses > misses_before


















def _with_unstuck(source: str, enabled: bool) -> str:
    if enabled:
        return source
    if "#if PASSIVBOT_UNSTUCK_ENABLED" not in source:
        raise RuntimeError("GPU source is missing the unstuck ablation contract")
    return "#define PASSIVBOT_UNSTUCK_ENABLED 0\n" + source


def _with_unstuck_ema(source: str, enabled: bool) -> str:
    if enabled:
        return source
    if "#if PASSIVBOT_UNSTUCK_EMA_ENABLED" not in source:
        raise RuntimeError("GPU source is missing the unstuck EMA ablation contract")
    return "#define PASSIVBOT_UNSTUCK_EMA_ENABLED 0\n" + source


def _with_multicoin_side(source, side):
    if side is None:
        return source
    if side not in {"long", "short"}:
        raise ValueError("multicoin compile side must be long, short or None")
    if "#ifdef PASSIVBOT_MULTICOIN_SHORT_SIDE" not in source:
        raise RuntimeError("GPU source is missing the multicoin side contract")
    return f"#define PASSIVBOT_MULTICOIN_SHORT_SIDE {int(side == 'short')}\n" + source


@lru_cache(maxsize=32)
def _ema_anchor_multicoin_shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    dynamic_wel_by_tradability: bool = True,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    hsl_disabled: bool = False,
    cuda_coin_capacity: int | None = None,
    hsl_capacity: int = 0,
    hsl_lookback: int = 0,
    mps_coin_capacity: int | None = None,
    raw_strategy_risk_enabled: bool = False,
    raw_strategy_growth_enabled: bool = False,
    hsl_raw_tail_capacity: int = 1,
    unstuck_ema_enabled: bool = True,
    hsl_fact_capacity: int = 0,
    hsl_factual_only: bool = False,
    unstuck_enabled: bool = True,
    side: str | None = None,
    temporal_chunking: bool = False,
    hsl_native_cache_capacity: int = 0,
    exposure_rms_enabled: bool = False,
    organillo_enabled: bool = False,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_multicoin_source_py(),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
    )
    source = (
        f"#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED {int(bool(hsl_native_cache_capacity))}\n"
        f"#define PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY {hsl_native_cache_capacity}\n"
        f"#define PASSIVBOT_EXPOSURE_RMS_ENABLED {int(exposure_rms_enabled)}\n"
        "#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n" + source
    )
    source = _with_unstuck_ema(source, unstuck_ema_enabled)
    source = _with_unstuck(source, unstuck_enabled)
    source = _with_multicoin_side(source, side)
    if temporal_chunking:
        if "#if PASSIVBOT_EMA_MULTICOIN_CHUNKED" not in source:
            raise RuntimeError("GPU source is missing the EMA replay-state contract")
        source = "#define PASSIVBOT_EMA_MULTICOIN_CHUNKED 1\n" + source
    # Keep diagnostics for forced delist panic-loss parity even when the HSL
    # controllers and their per-candle scans are compiled away.
    source = _with_hsl_disabled(source, hsl_disabled)
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    if raw_strategy_risk_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_RISK_ENABLED 1\n" + source
    if raw_strategy_growth_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_GROWTH_ENABLED 1\n" + source
    if organillo_enabled:
        source = "#define PASSIVBOT_ORGANILLO 1\n" + source
    source = _with_dynamic_wel_by_tradability(source, dynamic_wel_by_tradability)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    if hsl_capacity:
        source = f"#define PASSIVBOT_HSL_LOOKBACK {hsl_lookback}\n" + _with_hsl(
            source, hsl_capacity, hsl_fact_capacity
        )
    if mps_coin_capacity is not None:
        return compile_shader(
            source,
            cuda_coin_capacity=cuda_coin_capacity,
            mps_coin_capacity=mps_coin_capacity,
        )
    return compile_shader(source, cuda_coin_capacity=cuda_coin_capacity)


@lru_cache(maxsize=32)
def _trailing_martingale_multicoin_shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    dynamic_wel_by_tradability: bool = True,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    entry_interval_enabled: bool = False,
    temporal_chunking: bool = False,
    cuda_coin_capacity: int | None = None,
    hsl_capacity: int = 0,
    hsl_lookback: int = 0,
    mps_coin_capacity: int | None = None,
    loss_gate_disabled: bool = False,
    raw_strategy_risk_enabled: bool = False,
    raw_strategy_growth_enabled: bool = False,
    hsl_raw_tail_capacity: int = 1,
    unstuck_ema_enabled: bool = True,
    hsl_fact_capacity: int = 0,
    hsl_factual_only: bool = False,
    unstuck_enabled: bool = True,
    side: str | None = None,
    hsl_native_cache_capacity: int = 0,
    exposure_rms_enabled: bool = False,
    coin_overrides_absent: bool = False,
    hsl_native_coin_cache: bool = False,
    organillo_enabled: bool = False,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_hsl_features(
        _with_tm_loss_gate_disabled(
            passivbot_rust.mps_trailing_martingale_multicoin_source_py(), loss_gate_disabled
        ),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
    )
    source = (
        f"#define PASSIVBOT_COIN_OVERRIDES_ABSENT {int(coin_overrides_absent)}\n"
        f"#define PASSIVBOT_HSL_NATIVE_COIN_CACHE_ENABLED {int(hsl_native_coin_cache)}\n"
        f"#define PASSIVBOT_HSL_NATIVE_CACHE_ENABLED {int(bool(hsl_native_cache_capacity))}\n"
        f"#define PASSIVBOT_HSL_NATIVE_CACHE_CAPACITY {hsl_native_cache_capacity}\n"
        f"#define PASSIVBOT_EXPOSURE_RMS_ENABLED {int(exposure_rms_enabled)}\n"
        "#define PASSIVBOT_HSL_FACTUAL_ONLY 1\n" + source
    )
    source = _with_unstuck_ema(source, unstuck_ema_enabled)
    source = _with_unstuck(source, unstuck_enabled)
    source = _with_multicoin_side(source, side)
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    if raw_strategy_risk_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_RISK_ENABLED 1\n" + source
    if raw_strategy_growth_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_GROWTH_ENABLED 1\n" + source
    if organillo_enabled:
        source = "#define PASSIVBOT_ORGANILLO 1\n" + source
    source = _with_dynamic_wel_by_tradability(source, dynamic_wel_by_tradability)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    if temporal_chunking:
        if "#if PASSIVBOT_TM_MULTICOIN_CHUNKED" not in source:
            raise RuntimeError(
                "MPS source is missing the multicoin replay-state contract"
            )
        source = "#define PASSIVBOT_TM_MULTICOIN_CHUNKED 1\n" + source
    if hsl_capacity:
        source = f"#define PASSIVBOT_HSL_LOOKBACK {hsl_lookback}\n" + _with_hsl(
            source, hsl_capacity, hsl_fact_capacity
        )
    if mps_coin_capacity is not None:
        return compile_shader(
            source,
            cuda_coin_capacity=cuda_coin_capacity,
            mps_coin_capacity=mps_coin_capacity,
        )
    return compile_shader(source, cuda_coin_capacity=cuda_coin_capacity)


@lru_cache(maxsize=1)
def _strategy_eq_recovery_distribution_shader_library():
    gpu_device(torch)
    import passivbot_rust

    return compile_shader(
        passivbot_rust.mps_strategy_eq_recovery_distribution_source_py()
    )


def _strategy_eq_recovery_distribution_buffers(batch_size: int, sample_capacity: int, device):
    # Mutable reduction scratch belongs to this dispatch, never a process-global
    # cache shared by independent replay owners or CUDA streams. The allocator
    # reuses released storage without retaining inactive dataset histories here.
    shape = (int(batch_size), int(sample_capacity))
    return (
        torch.empty(shape, dtype=torch.int32, device=device),
        torch.empty(shape, dtype=torch.int32, device=device),
        torch.empty(
            (int(batch_size), MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(shape, dtype=torch.int32, device=device),
    )


def strategy_eq_recovery_distribution_from_samples(
    strategy_equity_samples, *, sample_interval_days: float = 1.0
):
    """Reduce strict time-to-exceed durations on uniformly spaced GPU samples."""

    if strategy_equity_samples.device.type not in {"mps", "cuda"}:
        raise ValueError(
            "strategy-equity recovery distribution requires an MPS or CUDA tensor"
        )
    if strategy_equity_samples.dtype != torch.float32:
        raise ValueError("strategy-equity recovery distribution requires float32 input")
    if strategy_equity_samples.ndim != 2:
        raise ValueError(
            "strategy-equity recovery distribution expects a batch-by-sample matrix"
        )
    sample_interval_days = float(sample_interval_days)
    if not np.isfinite(sample_interval_days) or sample_interval_days <= 0.0:
        raise ValueError("recovery sample interval must be finite and positive")
    matrix = strategy_equity_samples.contiguous()
    batch_size, sample_capacity = (int(value) for value in matrix.shape)
    if batch_size == 0:
        return torch.empty(
            (0, MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device=matrix.device,
        )
    if sample_capacity == 0:
        return torch.zeros(
            (batch_size, MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device=matrix.device,
        )
    stack, histogram, output, sizes = _strategy_eq_recovery_distribution_buffers(
        batch_size, sample_capacity, matrix.device
    )
    histogram.zero_()
    library = _strategy_eq_recovery_distribution_shader_library()
    library.passivbot_strategy_eq_recovery_distribution(
        matrix,
        stack,
        histogram,
        output,
        sizes,
        threads=(batch_size, 1, 1),
    )
    return output * sample_interval_days








def _recovery_history_bytes(sample_capacity: int):
    # Samples, a possible contiguous copy of a truncated view, the index stack
    # and duration histogram. Output and scaled result coexist until completion.
    return int(sample_capacity) * 16 + MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS * 8


class HslFactHistoryOverflow(RuntimeError):
    """A rejected GPU attempt may be repeated with larger worker-owned storage."""


def _metal_replay_group_options(kernel, batch_size):
    if gpu_device(torch) != "mps":
        return {}
    maximum = int(kernel.max_threads_per_threadgroup)
    width = int(kernel.thread_execution_width)
    if maximum < 1 or width < 1 or batch_size < 1:
        raise RuntimeError("Metal replay has invalid dispatch limits")
    # The API's implicit group may reserve private state for many inactive
    # lanes. Bound a state-heavy portfolio group by its actual candidates and
    # one execution width, retaining the compiled pipeline's limit as well.
    return {"group_size": (min(batch_size, maximum, width), 1, 1)}


def _require_available_held_valuation(scalars):
    # Scalar 9 normally holds -1 (not liquidated) or a liquidation day >= 0.
    # Metal writes -2 and returns immediately if a held coin has no price.
    if bool((scalars[:, 9] == -8.0).any()):
        raise ValueError("GPU v9 realized-loss memory or allowance received invalid risk inputs")
    if bool((scalars[:, 9] == -3.0).any()):
        raise RuntimeError("GPU fill-PnL history overflow")
    if bool((scalars[:, 9] == -5.0).any()):
        raise RuntimeError("GPU close-admission invariant failed")
    if bool((scalars[:, 9] == -7.0).any()):
        raise RuntimeError("GPU HSL malformed factual history")
    invalid_hsl = scalars[:, 9] == -4.0
    if bool(invalid_hsl.any()):
        indices = invalid_hsl.nonzero().flatten()
        rows = indices[:8].cpu().tolist()
        suffix = f" (+{indices.numel() - 8} more)" if indices.numel() > 8 else ""
        raise ValueError(
            f"GPU proxy unavailable HSL controller inputs; candidate rows {rows}{suffix}"
        )
    invalid = scalars[:, 9] == -2.0
    if bool(invalid.any()):
        indices = invalid.nonzero().flatten()
        rows = indices[:8].cpu().tolist()
        suffix = f" (+{indices.numel() - 8} more)" if indices.numel() > 8 else ""
        raise ValueError(
            "GPU proxy unavailable held-position valuation: candle outside its declared "
            "valid range or missing finite positive H/L/C; "
            f"candidate rows {rows}{suffix}"
        )

    # Capacity is recoverable only when every other candidate is valid. Never
    # retry a mixed batch containing an unavailable input or malformed fact.
    if bool((scalars[:, 9] == -6.0).any()):
        raise HslFactHistoryOverflow("GPU HSL factual history overflow")


def _decode_outputs(
    daily, scalars, gaps, *, btc_risk_enabled=False,
    native_factual_hsl=True, exposure_rms_enabled=False,
    interval_ms=60_000,
) -> dict:
    if native_factual_hsl is not True:
        raise ValueError("Retired GPU timestamp layout; current replay requires integer candle indices")
    expected = {62, 64, 66, 68, 67, 69, 71, 73}
    scalar_width = scalars.shape[1] - (7 if exposure_rms_enabled else 0) if scalars.ndim == 2 else -1
    if scalar_width not in expected:
        raise ValueError(f"Unsupported current GPU scalar layout: {tuple(scalars.shape)}")
    # Requested native RMS metrics append seven moments after the shared-account scalar tail.
    rms_scalars = scalars[:, -7:] if exposure_rms_enabled else None
    if exposure_rms_enabled:
        scalars = scalars[:, :-7]
    _require_available_held_valuation(scalars)
    active_days = torch.isfinite(daily[:, :, 1]) & (daily[:, :, 1] < float("inf"))

    def timestamp_column(index: int):
        values = scalars[:, index]
        return torch.where(values >= 0.0, values, torch.full_like(values, float("nan")))

    output = {
        "day_end_eq": daily[:, :, 0],
        "day_min_eq": torch.where(
            active_days,
            daily[:, :, 1],
            torch.full_like(daily[:, :, 1], float("inf")),
        ),
        "day_max_dd": daily[:, :, 2],
        "day_volume": daily[:, :, 3],
        "day_has_fill": daily[:, :, 4] > 0.0,
        "day_min_balance": torch.where(
            active_days,
            daily[:, :, 5],
            torch.full_like(daily[:, :, 5], float("inf")),
        ),
        "day_net_pnl": daily[:, :, 6],
        "day_last_fill_balance": daily[:, :, 7],
        "day_fill_count": daily[:, :, 8],
        "max_dd": scalars[:, 0],
        "held_max_ms": scalars[:, 1],
        "gap_hist": gaps,
        "gap_max_ms": scalars[:, 2],
        "first_fill_ts": timestamp_column(3),
        "last_fill_ts": timestamp_column(4),
        "recovery_max_ms": scalars[:, 5],
        "last_high_ts": timestamp_column(6),
        "first_eq_ts": timestamp_column(7),
        "last_eq_ts": timestamp_column(8),
        "liq_step": scalars[:, 9].to(torch.int64),
        "balance": scalars[:, 10],
        "psize": scalars[:, 11],
        "pprice": scalars[:, 12],
        "alive": scalars[:, 13] > 0.0,
        "open_positions": scalars[:, 14],
        "short_psize": scalars[:, 15],
        "short_pprice": scalars[:, 16],
        "profit_sum": scalars[:, 18],
        "loss_sum": scalars[:, 19],
        "position_unchanged_max_ms": scalars[:, 20],
        "entry_initial_balance_pct": scalars[:, 21],
        "total_wallet_exposure_max": scalars[:, 22],
        "total_wallet_exposure_mean": scalars[:, 23],
        "fill_count": scalars[:, 24],
        "fill_count_entry": scalars[:, 25],
        "fill_count_long": scalars[:, 26],
        "fills_active_days_count": scalars[:, 27],
        "pnl_recovery_max_ms": scalars[:, 28],
        "held_sum_ms": scalars[:, 29],
        "held_sum_squared_hours": scalars[:, -2],
        "gap_sum_squared_hours": scalars[:, -1],
        "held_count": scalars[:, 30],
        "account_recovery_max_ms": scalars[:, 31],
        "hsl_long_enabled": scalars[:, 32] > 0.0,
        "hsl_short_enabled": scalars[:, 33] > 0.0,
        "hsl_triggers_long": scalars[:, 34],
        "hsl_triggers_short": scalars[:, 35],
        "hsl_restarts_long": scalars[:, 36],
        "hsl_restarts_short": scalars[:, 37],
        # The HSL controller stores elapsed steps in the compact ABI.
        "hsl_tier_samples_total": scalars[:, 38],
        "hsl_tier_samples_red": scalars[:, 39],
        "hsl_duration_sum_steps": scalars[:, 40],
        "hsl_duration_max_steps": scalars[:, 41],
        "hsl_duration_count": scalars[:, 42],
        "hsl_trigger_drawdown_sum": scalars[:, 43],
        "hsl_trigger_drawdown_count": scalars[:, 44],
        "hsl_flatten_time_sum_steps": scalars[:, 45],
        "hsl_flatten_time_count": scalars[:, 46],
        "hsl_restart_retrigger_count": scalars[:, 47],
        "hsl_halt_to_restart_equity_loss": scalars[:, 48],
        "hsl_panic_close_loss_sum": scalars[:, 49],
        "hsl_panic_close_loss_max": scalars[:, 50],
        "hsl_panic_loss_drawdown_min": scalars[:, 51],
        "hsl_panic_loss_drawdown_sum": scalars[:, 52],
        "hsl_panic_loss_drawdown_max": scalars[:, 53],
        "hsl_panic_loss_drawdown_count": scalars[:, 54],
        "hsl_drawdown_ema_max_long": scalars[:, 55],
        "hsl_drawdown_ema_max_short": scalars[:, 56],
        "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 57],
        "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 58],
        "hsl_drawdown_ema_mean_worst_1pct_portfolio": scalars[:, -3],
        "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 59, reserved_columns=3
        ),
        "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 60, reserved_columns=3
        ),
        "hsl_drawdown_raw_max_long": _scalar_column_or_zero(
            scalars, 61, reserved_columns=3
        ),
        "hsl_drawdown_raw_max_short": _scalar_column_or_zero(
            scalars, 62, reserved_columns=3
        ),
        "hsl_drawdown_raw_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 63, reserved_columns=3
        ),
        "hsl_drawdown_raw_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 64, reserved_columns=3
        ),
    }
    if exposure_rms_enabled:
        for i, name in enumerate((
            "total_wallet_exposure_rms_long", "total_wallet_exposure_rms_short",
            "total_wallet_exposure_rms_gross", "total_wallet_exposure_utilization_rms_long",
            "total_wallet_exposure_utilization_rms_short", "position_exposure_utilization_rms_long",
            "position_exposure_utilization_rms_short",
        )):
            output[name] = rms_scalars[:, i]
    # Preserve exact indices on-device; host reduction restores float64 clocks.
    for name, column in (("first_eq", 7), ("last_eq", 8)):
        output[f"{name}_step"] = scalars[:, column].to(torch.int64)
        output[f"{name}_ts"] = timestamp_column(column) * float(interval_ms)
    # Column 17 owns the portfolio maximum; unified side signals are zero.
    output["hsl_drawdown_ema_max_portfolio"] = scalars[:, 17]
    if btc_risk_enabled:
        output.update(_decode_btc_risk_outputs(daily, active_days, 9))
    return output


def _decode_multicoin_fused_outputs(
    daily, scalars, gaps, *, btc_risk_enabled=False,
    native_factual_hsl=True, exposure_rms_enabled=False,
    interval_ms=60_000,
) -> dict:
    width = scalars.shape[1] - (7 if exposure_rms_enabled else 0) if scalars.ndim == 2 else -1
    if width not in {67, 69, 71, 73}:
        raise ValueError(f"Unsupported current fused GPU scalar layout: {tuple(scalars.shape)}")
    output = _decode_outputs(daily, scalars, gaps, btc_risk_enabled=btc_risk_enabled,
                             native_factual_hsl=native_factual_hsl,
                             exposure_rms_enabled=exposure_rms_enabled, interval_ms=interval_ms)
    if exposure_rms_enabled:
        scalars = scalars[:, :-7]
    long_entry_initial_balance_pct = output.pop("entry_initial_balance_pct")
    output.update(
        {
            "entry_initial_balance_pct_long": long_entry_initial_balance_pct,
            "entry_initial_balance_pct_short": scalars[:, 57],
            "profit_sum_long": scalars[:, 58],
            "loss_sum_long": scalars[:, 59],
            "profit_sum_short": scalars[:, 60],
            "loss_sum_short": scalars[:, 61],
            "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 62],
            "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 63],
            "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
                scalars, 64, reserved_columns=3
            ),
            "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
                scalars, 65, reserved_columns=3
            ),
            "hsl_drawdown_raw_max_long": _scalar_column_or_zero(
                scalars, 66, reserved_columns=3
            ),
            "hsl_drawdown_raw_max_short": _scalar_column_or_zero(
                scalars, 67, reserved_columns=3
            ),
            "hsl_drawdown_raw_mean_worst_1pct_long": _scalar_column_or_zero(
                scalars, 68, reserved_columns=3
            ),
            "hsl_drawdown_raw_mean_worst_1pct_short": _scalar_column_or_zero(
                scalars, 69, reserved_columns=3
            ),
        }
    )
    return output




def _validate_global_memory_params(params, keys, *, sides):
    """Global request policies may vary by candidate, never by side or coin."""
    from optimization.gpu.hsl import validate_memory_span

    width = len(keys)
    for key in ("unstuck_memory_ema_span_minutes", "loss_gate_memory_ema_span_minutes"):
        column = keys.index(key)
        values = params[:, column]
        for value in np.unique(values):
            validate_memory_span(value)
        for side in range(1, sides):
            if np.any(values != params[:, side * width + column]):
                raise ValueError(f"GPU {key} must be account-global")


def _validate_hsl_params(runner, params, keys, *, effective_enabled=None):
    width = params.shape[1] // 2
    for side, active in enumerate((runner.long_enabled, runner.short_enabled)):
        if not active:
            continue
        cols = {
            key: params[:, side * width + keys.index(key)]
            for key in (
                "hsl_enabled",
                "hsl_red_threshold",
                "hsl_drawdown_ratio_ema_span_minutes",
                "hsl_cooldown_ms_lo",
                "hsl_memory_ema_span_minutes",
                "hsl_signal_mode",
                "hsl_cooldown_ms_hi",
            )
        }
        enabled = cols["hsl_enabled"] > 0.5
        if not np.isfinite(cols["hsl_enabled"]).all():
            raise ValueError("GPU HSL enablement must be finite")
        # HSL clocks belong to the selected scope, independently of realized memory.
        for key in ("hsl_memory_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes"):
            spans = cols[key].astype(np.float32)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                alpha = np.float32(2) / (spans + np.float32(1))
                beta = np.float32(1) - alpha
            if np.any(~np.isfinite(spans) | (spans < 1) | ~np.isfinite(alpha)
                      | (alpha <= 0) | (beta < 0) | (beta >= 1)):
                raise ValueError(f"GPU {key} cannot encode the shared f32 decay")
        if not np.any(enabled):
            continue
        needs_history = (enabled if effective_enabled is None
                         else enabled & effective_enabled[:, side])
        if np.any(needs_history) and runner.interval_minutes != 1:
            raise ValueError("Enabled v9 GPU HSL requires 1m candles")
        if any(not np.isfinite(v[enabled]).all() for v in cols.values()):
            raise ValueError("GPU HSL parameters must be finite")

        def selected(key):
            return cols[key][enabled]

        if (
            np.any(selected("hsl_red_threshold") <= 0)
            or np.any(selected("hsl_red_threshold") > 1)
            or np.any(selected("hsl_drawdown_ratio_ema_span_minutes") < 1)
            or np.any(selected("hsl_memory_ema_span_minutes") < 1)
            or not np.isin(selected("hsl_signal_mode"), [0, 1, 2]).all()
            or any(np.any(selected(key) < 0) or np.any(selected(key) >= 2**24)
                   or np.any(selected(key) != np.floor(selected(key)))
                   for key in ("hsl_cooldown_ms_lo", "hsl_cooldown_ms_hi"))
        ):
            raise ValueError("Invalid GPU HSL policy")
    # Both directional views must identify the same topology. Unified has
    # one explicitly packed policy; never choose between conflicting views.
    mode = keys.index("hsl_signal_mode")
    if np.any(params[:, mode] != params[:, width + mode]):
        raise ValueError("GPU HSL views disagree on signal mode")
    unified = params[:, mode] == 0
    for key in (
        "hsl_enabled",
        "hsl_red_threshold",
        "hsl_drawdown_ratio_ema_span_minutes",
        "hsl_cooldown_ms_lo",
        "hsl_cooldown_ms_hi",
        "hsl_memory_ema_span_minutes",
    ):
        i = keys.index(key)
        if np.any(params[unified, i] != params[unified, width + i]):
            raise ValueError("Unified GPU HSL requires one shared policy")




def _validate_market_execution_data(data, market_order_slippage_pct):
    """Reject stale preparation before any runner allocation or dispatch."""
    slippage = validate_market_order_slippage_pct(market_order_slippage_pct)
    required = {"market_execution_ticks", "market_execution_ticks_abi", "market_order_slippage_pct"}
    if not required <= data.keys():
        raise ValueError("Prepared GPU data lacks native market execution ticks")
    if (data["market_execution_ticks_abi"] != MARKET_EXECUTION_TICKS_ABI
            or data["market_order_slippage_pct"] != slippage):
        raise ValueError("Prepared GPU market execution ABI/slippage disagrees with runner")
    ticks = data["market_execution_ticks"]
    if (not isinstance(ticks, torch.Tensor)
            or tuple(ticks.shape) != (int(data["n"]), int(data["n_coins"]), 2)
            or ticks.dtype != torch.int32
            or ticks.device != data["bars"].device
            or not ticks.is_contiguous()):
        raise ValueError("Prepared GPU market execution ticks have invalid shape/dtype/device/layout")
    return ticks


def _pack_replay_control(sizes, end_steps, recovery_distribution_enabled):
    """Snapshot this attempt's sizes and stops in one O(batch) readonly input."""
    prefix_words = 10 if recovery_distribution_enabled else 8
    if (sizes.dtype != torch.int32 or end_steps.dtype != torch.int32
            or sizes.ndim != 1 or sizes.numel() != prefix_words
            or end_steps.ndim != 1 or sizes.device != end_steps.device):
        raise ValueError("Invalid GPU replay control sizes/end_steps layout")
    # Always repack: identical batch sizes can have different candidate stops,
    # including another factual attempt or a following partial replay.
    return torch.cat((sizes, end_steps)).contiguous()


class _MulticoinReplayRunner:
    """Worker-owned multicoin allocation, retry and dispatch lifecycle."""

    scalar_cols = MPS_MULTICOIN_SCALAR_COLS
    replay_sides = 1
    organillo_enabled = False
    organillo_buffers = ()
    side_specialization = True
    cuda_whole_group_size = None
    temporal_chunk_bar_limit = MPS_TM_MULTICOIN_CHUNK_BARS
    supports_entry_interval = False
    decimal_coin_overrides = False
    specialize_metal_coin_capacity = False
    _validate_hsl_params = _validate_hsl_params

    def _use_disabled_hsl_specialization(self, matrix):
        return False

    def _compiled_side(self):
        # Preserve the existing Metal path until its runtime is validated.
        if self.replay_sides == 1 and self.side_specialization and self.bars.device.type == "cuda":
            return self.side
        return None

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        side: str,
        coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        filter_by_min_effective_cost: dict[str, bool],
        side_activation: dict[str, bool],
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        compact_recovery_output: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        max_dispatch_candidate_bars: int | None = None,
        factual_hsl: bool = True,
        exposure_rms_enabled: bool = False,
        interrupt_check=None,
        selection=None,
    ):
        self.market_execution_ticks = _validate_market_execution_data(data, market_order_slippage_pct)
        if max_dispatch_candidate_bars is not None and max_dispatch_candidate_bars <= 0:
            raise ValueError("max_dispatch_candidate_bars must be positive")
        self.max_dispatch_candidate_bars = max_dispatch_candidate_bars
        self._replay_state_bytes = None
        self._replay_state_sizes = {}
        self._replay_states = {}
        self._last_temporal_dispatch = None
        if side not in {"long", "short"}:
            raise ValueError(
                f"MPS multicoin runner side must be long or short, got {side!r}"
            )
        self.side = side
        self.side_activation = validate_side_activation(side_activation)
        if factual_hsl is not True:
            raise ValueError("Retired GPU replay layout; current shared-account replay requires factual_hsl=True")
        self.native_factual_hsl = True
        if self.native_factual_hsl and int(data["n"]) > 2**24:
            raise ValueError("Native GPU replay exceeds exact candle-index capacity")
        self.exposure_rms_enabled = bool(exposure_rms_enabled)
        if self.exposure_rms_enabled and not self.native_factual_hsl:
            raise ValueError("Exposure RMS GPU metrics require native CUDA replay")
        self.interrupt_check = interrupt_check or (lambda: None)
        self.collect_coin_fill_counts = bool(collect_coin_fill_counts)
        self.hsl_ema_tail_enabled = bool(hsl_ema_tail_enabled)
        self.hsl_raw_drawdown_enabled = bool(hsl_raw_drawdown_enabled)
        self.hsl_raw_tail_enabled = bool(hsl_raw_tail_enabled)
        self.recovery_distribution_enabled = bool(recovery_distribution_enabled)
        self.compact_recovery_output = bool(compact_recovery_output)
        self.raw_strategy_risk_enabled = bool(raw_strategy_risk_enabled)
        self.raw_strategy_growth_enabled = bool(raw_strategy_growth_enabled)
        self.dynamic_wel_by_tradability = bool(dynamic_wel_by_tradability)
        fused = self.replay_sides == 2
        self.long_enabled = fused or side == "long"
        self.short_enabled = fused or side == "short"
        if self.hsl_raw_tail_enabled:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_SCALAR_COLS if fused else MPS_MULTICOIN_SCALAR_COLS
            )
        elif self.hsl_raw_drawdown_enabled:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_RAW_DRAWDOWN_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_RAW_DRAWDOWN_SCALAR_COLS
            )
        elif self.hsl_ema_tail_enabled:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_EMA_TAIL_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_EMA_TAIL_SCALAR_COLS
            )
        else:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_BASE_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_BASE_SCALAR_COLS
            )
        if self.exposure_rms_enabled:
            self.scalar_cols += 7
        self.run_config = run
        interval_minutes = float(run.interval_ms) / 60_000.0
        if (
            not np.isfinite(interval_minutes)
            or interval_minutes < 1.0
            or not interval_minutes.is_integer()
        ):
            raise ValueError(
                "MPS multicoin runner requires a positive whole-minute candle interval"
            )
        self.interval_minutes = int(interval_minutes)
        self.n = int(data["n"])
        self.n_coins = int(data["n_coins"])
        self.n_days = int(data["n_days"])
        self.hsl_raw_tail_capacity = (
            _raw_drawdown_tail_capacity(self.n_days) if self.hsl_raw_tail_enabled else 1
        )
        self.pnl_lookback_bars = int(pnl_lookback_bars)
        self.hsl_capacity = 0
        self._hsl_scratch_buffers = {}
        self.hsl_scratch_budget_bytes = 512 * 1024 * 1024
        self.hsl_replay_sides = self.replay_sides
        self.hsl_scopes = self.hsl_replay_sides * (self.n_coins + 1)
        # Conservative resident seed; completed overflow learns upward. Keep
        # that estimate across dispatches where the effective HSL policy is off.
        self.hsl_fact_capacity_learned = 0
        self.hsl_fact_capacity = 0
        self.hsl_factual_replay = False
        if pnl_lookback_bars != 0 and (
            self.interval_minutes != 1
        ):
            raise ValueError("v9 GPU HSL requires 1m candles")
        self.hsl_capacity = 2

        self.btc_prices = _btc_risk_price_tensor(btc_prices, expected_count=self.n)
        self.equity_balance_diff_enabled = bool(equity_balance_diff_enabled)
        self.entry_interval_enabled = bool(entry_interval_enabled)
        if self._exact_entry_intervals_enabled():
            entry_interval_history_bytes(self.n)
        if (
            self.entry_interval_enabled
            and not self.supports_entry_interval
        ):
            raise ValueError(
                "MPS entry-interval output is only defined for Trailing Martingale"
            )
        self.btc_risk_enabled = (
            self.btc_prices is not None
            if btc_risk_enabled is None
            else bool(btc_risk_enabled)
        )
        if (
            self.btc_risk_enabled or self.equity_balance_diff_enabled
        ) and self.btc_prices is None:
            raise ValueError("MPS opt-in BTC-priced metrics require BTC prices")
        self.btc_prices_enabled = (
            self.btc_risk_enabled or self.equity_balance_diff_enabled
        )
        self.daily_cols = (
            MPS_MULTICOIN_DAILY_COLS + (3 if self.btc_risk_enabled else 0)
            + int(self.raw_strategy_risk_enabled)
            + 2 * int(self.raw_strategy_growth_enabled)
        )
        self.recovery_stride = 1 if self.recovery_distribution_enabled else 0
        self.n_recovery_samples = (
            max(
                1,
                (self.n + self.recovery_stride - 1) // self.recovery_stride + 1,
            )
            if self.recovery_distribution_enabled
            else 1
        )
        self.bars = data["bars"]
        self.cuda_coin_capacity = None
        self.mps_coin_capacity = None
        if (
            self.hsl_capacity
            and self.bars.device.type == "mps"
            and (self.specialize_metal_coin_capacity or self.native_factual_hsl)
        ):
            if not 1 <= self.n_coins <= MPS_MULTICOIN_MAX_COINS:
                raise ValueError("Metal coin count exceeds the multicoin shader limit")
            self.mps_coin_capacity = 1 << (self.n_coins - 1).bit_length()
        if self.bars.device.type == "cuda":
            if not 1 <= self.n_coins <= MPS_MULTICOIN_MAX_COINS:
                raise ValueError("CUDA coin count exceeds the multicoin shader limit")
            # Bound CUDA private arrays while sharing compiled variants within buckets.
            self.cuda_coin_capacity = 1 << (self.n_coins - 1).bit_length()
        self.fill_ticks = data["fill_ticks"]
        self.touch_ticks = data["touch_ticks"]
        self.touch_nearest_ticks = data["touch_nearest_ticks"]
        self.touch_min_qty_bits = data["touch_min_qty_bits"]
        self.touch_min_qty_relation = data["touch_min_qty_relation"]
        self.hour_log_ranges = data["hour_log_ranges"]
        self.coin_settings = data["coin_settings"]
        self._hourly_first_observed = data.get("hourly_first_observed")
        self._hourly_coin_ranges = data.get("hourly_coin_ranges")
        self.organillo_enabled = selection is not None
        self.organillo_buffers = ()
        if selection is not None:
            from organillo import validate_buffers
            validate_buffers(selection.values, selection.rows, self.n, self.n_coins)
            # Immutable data is shared by directional/fused runners and reused on residency.
            key = tuple((key, str(value)) for key, value in selection.identity.items())
            cache = data.setdefault("organillo_buffers", {})
            if key not in cache:
                cache.clear()
                cache[key] = (torch.tensor(selection.values, dtype=torch.uint8, device=gpu_device()),
                              torch.tensor(selection.rows, dtype=torch.int32, device=gpu_device()))
            self.organillo_buffers = cache[key]
        if coin_overrides is None:
            coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols), np.nan, dtype=np.float32
            )
        coin_overrides = np.asarray(coin_overrides, dtype=np.float64)
        if coin_overrides.shape != (self.n_coins, self.coin_override_cols):
            raise ValueError(
                f"expected multicoin {self.coin_override_label} override matrix shaped "
                f"({self.n_coins}, {self.coin_override_cols}), "
                f"got {coin_overrides.shape}"
            )
        self._bootstrap_coin_overrides = (np.array(coin_overrides, dtype=np.float64, copy=True),)
        wel_column = self.coin_override_wel_column
        self.rms_ranking_coin_counts = (int(np.count_nonzero(coin_overrides[:, wel_column] != 0.0)),)
        cooldown_column = self.coin_override_cooldown_column
        self.rms_cooldown_coin_overrides = (coin_overrides[:, [
            cooldown_column, self.coin_override_cols - 4, self.coin_override_cols - 3,
            self.coin_override_cols - 1, wel_column,
        ]].copy(),)
        hsl_start = self.coin_override_hsl_start
        hsl_overrides = coin_overrides[:, hsl_start]
        self._coin_hsl_enabled_overrides = (hsl_overrides.copy(),)
        self.coin_hsl_may_enable = bool(
            np.any(np.isfinite(hsl_overrides) & (hsl_overrides > 0.5))
        )
        unstuck_start = self.coin_override_unstuck_start
        self._unstuck_ema_overrides = (coin_overrides[:, [unstuck_start, unstuck_start + 1]].copy(),)
        prepared_overrides = self._prepare_coin_overrides(coin_overrides)
        # Only immutable dataset values specialize lookup; candidate fallbacks
        # stay in the Rust-owned shader and may differ on every dispatch.
        self.coin_overrides_absent = not bool(np.isfinite(coin_overrides).any())
        self.coin_overrides = torch.as_tensor(prepared_overrides, device=gpu_device())
        forager_score_hysteresis_pct = float(forager_score_hysteresis_pct)
        if not np.isfinite(forager_score_hysteresis_pct) or (
            forager_score_hysteresis_pct < 0.0
        ):
            raise ValueError(
                "forager_score_hysteresis_pct must be finite and non-negative"
            )
        max_realized_loss_pct = float(max_realized_loss_pct)
        if not np.isfinite(max_realized_loss_pct) or max_realized_loss_pct < 0.0:
            raise ValueError("max_realized_loss_pct must be finite and non-negative")
        self.loss_gate_enabled = max_realized_loss_pct < 1.0
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        market_order_slippage_pct = float(market_order_slippage_pct)
        if (
            not np.isfinite(market_order_slippage_pct)
            or market_order_slippage_pct < 0.0
        ):
            raise ValueError(
                "market_order_slippage_pct must be finite and non-negative"
            )
        market_order_near_touch_threshold = float(market_order_near_touch_threshold)
        if (
            not np.isfinite(market_order_near_touch_threshold)
            or market_order_near_touch_threshold < 0.0
        ):
            raise ValueError(
                "market_order_near_touch_threshold must be finite and non-negative"
            )
        liq_floor = max(0.0, run.starting_balance) * max(0.0, run.liquidation_threshold)
        minimum_cost_filters = validate_minimum_cost_filters(filter_by_min_effective_cost)
        self.settings = torch.tensor(
            [
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                float(side == "short"),
                forager_score_hysteresis_pct,
                encoded_max_realized_loss_pct,
                float(self.collect_coin_fill_counts),
                market_order_slippage_pct,
                float(bool(hsl_panic_market)),
                float(bool(market_orders_allowed)),
                market_order_near_touch_threshold,
                float(minimum_cost_filters[side]),
                0.0,  # Runtime factual-ring capacity; absent from compile identity.
                0.0,  # Internal factual evaluation control, independently of capture.
                float(self.loss_gate_enabled),  # Original float64 gate activation.
                float(self.side_activation[side]),  # Directional activation ABI 1.
            ],
            dtype=torch.float32,
            device=gpu_device(),
        )
        self._buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self._recovery_buffers: dict[int, torch.Tensor] = {}
        self._hsl_ema_tail_buffers: dict[int, torch.Tensor] = {}
        self._equity_balance_diff_buffers: dict[int, torch.Tensor] = {}
        self._entry_interval_stat_buffers: dict[int, torch.Tensor] = {}
        self._entry_interval_count_buffers: dict[int, torch.Tensor] = {}
        self._sizes: dict[tuple[int, int], torch.Tensor] = {}
        self._full_end_steps: dict[int, torch.Tensor] = {}
        self.last_profile: dict[str, float | int | bool] = {}
        self.start_minute_of_day = int(data["start_minute_of_day"])
        self.start_minute_of_hour = int(data["start_minute_of_hour"])
        self.requested_start_idx = max(
            0,
            int(
                (run.guard_ts_ms - int(data["ts0"]) + run.interval_ms - 1)
                // run.interval_ms
            ),
        )

    def _output_buffers(self, batch_size: int):
        if batch_size not in self._buffers:
            self._buffers = {
                batch_size: (
                    torch.zeros(
                        (batch_size, self.n_days, self.daily_cols),
                        dtype=torch.float32,
                        device=gpu_device(),
                    ),
                    torch.zeros(
                        (batch_size, self.scalar_cols),
                        dtype=torch.float32,
                        device=gpu_device(),
                    ),
                    torch.zeros(
                        (batch_size, GAP_BINS), dtype=torch.int32, device=gpu_device()
                    ),
                    torch.zeros(
                        (
                            (batch_size, self.n_coins)
                            if self.collect_coin_fill_counts
                            else (1,)
                        ),
                        dtype=torch.float32,
                        device=gpu_device(),
                    ),
                )
            }
        else:
            for buffer in self._buffers[batch_size]:
                buffer.zero_()
        self._buffers[batch_size][0][:, :, 1].fill_(float("inf"))
        self._buffers[batch_size][0][:, :, 5].fill_(float("inf"))
        if self.raw_strategy_growth_enabled:
            offset = self.daily_cols - int(self.raw_strategy_risk_enabled) - 1
            self._buffers[batch_size][0][:, :, offset].fill_(float("inf"))
        return self._buffers[batch_size]

    def _end_steps(self, end_steps: np.ndarray | None, batch_size: int):
        if end_steps is None:
            if batch_size not in self._full_end_steps:
                self._full_end_steps[batch_size] = torch.full(
                    (batch_size,), self.n - 1, dtype=torch.int32, device=gpu_device()
                )
            return self._full_end_steps[batch_size]
        values = effective_end_steps(end_steps, candidates=batch_size, available=self.n)
        return torch.as_tensor(
            np.ascontiguousarray(values), dtype=torch.int32, device=gpu_device()
        )

    def _library(self):
        loader, args = self._library_cache_call()
        return loader(*args)

    def _unstuck_required(self, matrix):
        if not getattr(self, "unstuck_specialization", True):
            return True
        return unstuck_required(matrix, self.parameter_keys, self._unstuck_ema_overrides)

    def _unstuck_ema_required(self, matrix):
        if not getattr(self, "unstuck_ema_specialization", True):
            return True
        keys = self.parameter_keys
        return unstuck_ema_required(matrix, keys, self._unstuck_ema_overrides)

    # Disposable unified factual arithmetic. Strategy subclasses enable only
    # supported device shapes; this internal flag retains scalar ablation.
    hsl_native_cache_enabled = False
    hsl_native_coin_cache_enabled = False
    hsl_native_cache_budget_bytes = 64 * 1024 * 1024

    def _hsl_native_coin_cache_selected(self):
        return bool(
            getattr(self, "hsl_native_coin_cache_enabled", False)
            and getattr(self, "dispatch_native_hsl_coin_cache_eligible", False)
            and self.bars.device.type in {"cuda", "mps"}
        )

    def _hsl_native_cache_layout_bytes(self, capacity):
        if self._hsl_native_coin_cache_selected():
            # Coin controllers own independent reconstructed pairs and events.
            return (_hsl_native_cache_bytes(capacity, 1, self.hsl_fact_capacity)
                    * self.hsl_replay_sides * self.n_coins)
        return _hsl_native_cache_bytes(
            capacity, self.hsl_replay_sides * self.n_coins, self.hsl_fact_capacity
        )

    def _hsl_native_cache_capacity(self):
        return 0

    def _disable_native_cache_over_budget(self):
        # Optional arithmetic must not prevent the authoritative histories from
        # fitting. Latch off for this request/attempt before selecting its ABI;
        # changing compiled state size must not oscillate cache admission.
        if (self._hsl_native_cache_capacity()
                and self._history_batch_bytes(1) > self.hsl_scratch_budget_bytes):
            self.dispatch_native_hsl_cache_budget_disabled = True
            return True
        return False

    def _hsl_native_cache_bytes_per_candidate(self):
        capacity = self._hsl_native_cache_capacity()
        return self._hsl_native_cache_layout_bytes(capacity) if capacity else 0

    def _hsl_history_bytes_per_candidate(self):
        if not self.hsl_capacity:
            return 0
        factual_only = getattr(self, "native_factual_hsl", True)
        _, nodes = _hsl_layout(self.hsl_capacity, getattr(self, "hsl_fact_capacity", 0),
                               factual_only=factual_only)
        return (self.hsl_scopes * (nodes * 32 + (0 if factual_only else self.hsl_capacity * 8))
                + self._hsl_native_cache_bytes_per_candidate())

    def _hsl_ema_tail_scopes(self):
        return ("long", "short", "portfolio") if self.replay_sides == 2 else (self.side, "portfolio")

    def _exact_entry_intervals_enabled(self):
        return bool(getattr(self, "native_factual_hsl", True)
                    and getattr(self, "entry_interval_enabled", False))

    def _hsl_ema_tail_samples_enabled(self):
        return bool(getattr(self, "native_factual_hsl", True)
                    and getattr(self, "hsl_ema_tail_enabled", False))

    def _history_bytes_per_candidate(self):
        return (
            4  # Fresh packed per-candidate end_steps word.
            + self._hsl_history_bytes_per_candidate()
            + 4 * BOOTSTRAP_COUNT_COLS * self.replay_sides * self.n_coins
            + (int(self._replay_state_bytes or 0) if self._temporal_replay_enabled() else 0)
            + (4 * self.n_days if self.raw_strategy_risk_enabled else 0)
            + (8 * self.n_days if self.raw_strategy_growth_enabled else 0)
            + (_recovery_history_bytes(self.n_recovery_samples)
               if self.recovery_distribution_enabled else 0)
            + (drawdown_ema_tail_history_bytes(self.n, len(self._hsl_ema_tail_scopes()))
               if self._hsl_ema_tail_samples_enabled() else 0)
            + (entry_interval_history_bytes(self.n)
               if self._exact_entry_intervals_enabled() else 0)
        )

    def _history_workspace_bytes(self):
        return 4 * (10 if self.recovery_distribution_enabled else 8)

    def _history_batch_bytes(self, batch_size):
        return batch_size * self._history_bytes_per_candidate() + self._history_workspace_bytes()

    def _prepare_scratch_shape(self, batch_size):
        """Drop completed automatic CUDA scratch before allocating a new shape."""
        if not getattr(self, "automatic_scratch_budget", False):
            return
        shape = (batch_size, self.hsl_fact_capacity,
                 self._hsl_native_cache_bytes_per_candidate(), self._replay_state_bytes)
        if shape == getattr(self, "_scratch_shape", None):
            return
        # Clearing one buffer at its allocator still overlaps the new tensor
        # with every other old history. Release the complete prior dispatch
        # before params, outputs or any replacement histories are allocated.
        for name in (
            "_buffers", "_hsl_scratch_buffers",
            "_recovery_buffers",
            "_hsl_ema_tail_buffers", "_equity_balance_diff_buffers",
            "_entry_interval_stat_buffers", "_entry_interval_count_buffers",
            "_replay_states", "_sizes", "_full_end_steps",
        ):
            getattr(self, name).clear()
        self._scratch_shape = shape

    def _history_batch_limit(self):
        resident = self._history_bytes_per_candidate()
        available = self.hsl_scratch_budget_bytes - self._history_workspace_bytes()
        return available // max(1, resident)

    def _hsl_buffers(self, batch_size):
        factual_only = getattr(self, "native_factual_hsl", True)
        _, nodes = _hsl_layout(self.hsl_capacity, getattr(self, "hsl_fact_capacity", 0),
                               factual_only=factual_only)
        if (
            self._history_batch_bytes(batch_size)
            > self.hsl_scratch_budget_bytes
        ):
            raise ValueError("GPU history batch exceeds its scratch budget")
        cache_bytes = self._hsl_native_cache_bytes_per_candidate()
        row_shape = ((batch_size, cache_bytes // 4) if cache_bytes else
                     (batch_size, self.hsl_scopes, 0 if factual_only else self.hsl_capacity * 2))
        key = (batch_size, getattr(self, "hsl_fact_capacity", 0), cache_bytes)
        if key not in self._hsl_scratch_buffers:
            self._hsl_scratch_buffers = {
                key: (
                    torch.empty(
                        (batch_size, self.hsl_scopes, nodes, 32),
                        dtype=torch.uint8,
                        device=gpu_device(),
                    ),
                    torch.empty(
                        row_shape,
                        dtype=torch.int32,
                        device=gpu_device(),
                    ),
                )
            }
        return self._hsl_scratch_buffers[key]

    def _run_history_batches(self, params, *, profile, end_steps):
        limit = self._history_batch_limit()
        if limit < 1:
            raise ValueError("GPU history exceeds the scratch budget")
        if len(params) <= limit:
            return None
        outputs, profiles = [], []
        for start in range(0, len(params), limit):
            result = self.run(
                params[start : start + limit],
                profile=profile,
                end_steps=(
                    None if end_steps is None else end_steps[start : start + limit]
                ),
            )
            outputs.append(
                {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in result.items()
                }
            )
            profiles.append(dict(self.last_profile))
            del result
        combined = {}
        for key, value in outputs[0].items():
            if isinstance(value, torch.Tensor):
                combined[key] = torch.cat([o[key] for o in outputs], dim=0)
            else:
                if any(o[key] != value for o in outputs):
                    raise ValueError(f"GPU sub-batches disagree on {key}")
                combined[key] = value
        self.last_profile = {}
        if profile:
            self.last_profile = {
                key: sum(p.get(key, 0) for p in profiles)
                for key in profiles[0]
                if key.endswith("_seconds")
            }
            self.last_profile.update(
                batch_size=len(params),
                candidate_batch_count=len(outputs),
                dispatch_count=sum(p.get("dispatch_count", 1) for p in profiles),
                kernel_candidate_steps=sum(
                    p["kernel_candidate_steps"] for p in profiles
                ),
                candidate_batch_sizes=[p["batch_size"] for p in profiles],
                cold_dispatch_count=sum(int(p.get("cold", False)) for p in profiles),
                cold=any(p.get("cold", False) for p in profiles),
            )
        return combined

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled,
                             native_factual_hsl=getattr(self, "native_factual_hsl", True),
                             exposure_rms_enabled=getattr(self, "exposure_rms_enabled", False),
                             interval_ms=self.run_config.interval_ms)

    def _hsl_ema_tail_sample_buffer(self, batch_size):
        if not self._hsl_ema_tail_samples_enabled():
            self._hsl_ema_tail_buffers.clear()
            return None
        if self._history_batch_bytes(batch_size) > self.hsl_scratch_budget_bytes:
            raise ValueError("GPU HSL EMA tail exceeds its scratch budget")
        if batch_size not in self._hsl_ema_tail_buffers:
            self._hsl_ema_tail_buffers = {
                batch_size: torch.empty(
                    (batch_size, len(self._hsl_ema_tail_scopes()), self.n),
                    dtype=torch.float32, device=gpu_device(),
                )
            }
        samples = self._hsl_ema_tail_buffers[batch_size]
        # Reset on every physical attempt, including retries and partial replays.
        samples.fill_(float("nan"))
        return samples

    def _recovery_sample_buffer(self, batch_size: int):
        if self._history_batch_bytes(batch_size) > self.hsl_scratch_budget_bytes:
            raise ValueError("GPU history batch exceeds its scratch budget")
        if batch_size not in self._recovery_buffers:
            self._recovery_buffers = {
                batch_size: torch.full(
                    (batch_size, self.n_recovery_samples),
                    float("nan"),
                    dtype=torch.float32,
                    device=gpu_device(),
                )
            }
        else:
            self._recovery_buffers[batch_size].fill_(float("nan"))
        return self._recovery_buffers[batch_size]




    def _equity_balance_diff_buffer(self, batch_size: int):
        if not self.equity_balance_diff_enabled:
            return None
        if batch_size not in self._equity_balance_diff_buffers:
            self._equity_balance_diff_buffers = {
                batch_size: torch.zeros(
                    (batch_size, MPS_EQUITY_BALANCE_DIFF_COLS),
                    dtype=torch.float32,
                    device=gpu_device(),
                )
            }
        else:
            self._equity_balance_diff_buffers[batch_size].zero_()
        return self._equity_balance_diff_buffers[batch_size]

    def _entry_interval_buffers(self, batch_size: int):
        if not self.entry_interval_enabled:
            return None, None
        count_cols = MPS_ENTRY_INTERVAL_COUNT_COLS
        if self._exact_entry_intervals_enabled():
            if self._history_batch_bytes(batch_size) > self.hsl_scratch_budget_bytes:
                raise ValueError("native entry intervals exceed the scratch budget")
            count_cols = self.n + 2
            if batch_size * count_cols > np.iinfo(np.int32).max:
                raise ValueError("native entry interval buffer exceeds integer addressing bounds")
        if batch_size not in self._entry_interval_stat_buffers:
            self._entry_interval_stat_buffers = {
                batch_size: torch.zeros(
                    (batch_size, MPS_ENTRY_INTERVAL_STAT_COLS),
                    dtype=torch.float32,
                    device=gpu_device(),
                )
            }
            self._entry_interval_count_buffers = {
                batch_size: torch.zeros(
                    (batch_size, count_cols),
                    dtype=torch.int32,
                    device=gpu_device(),
                )
            }
        else:
            self._entry_interval_stat_buffers[batch_size].zero_()
            self._entry_interval_count_buffers[batch_size].zero_()
        return (
            self._entry_interval_stat_buffers[batch_size],
            self._entry_interval_count_buffers[batch_size],
        )

    def _native_hsl_effective_enablement(self, matrix, keys):
        modes = keys.index("hsl_signal_mode")
        enables = keys.index("hsl_enabled")
        effective = []
        for side, overrides in enumerate(self._coin_hsl_enabled_overrides):
            offset = side * len(keys)
            enabled = matrix[:, enables + offset] > 0.5
            coin_mode = matrix[:, modes + offset] == 2
            explicit = np.isfinite(overrides)
            override_on = bool(np.any(explicit & (overrides > 0.5)))
            inherited = bool(np.any(~explicit))
            effective.append(np.where(coin_mode, override_on | (inherited & enabled), enabled))
        return np.stack(effective, axis=1)

    def _prepare_native_factual_hsl(self, params):
        self.dispatch_native_hsl_cache_eligible = False
        self.dispatch_native_hsl_coin_cache_eligible = False
        self.dispatch_native_hsl_cache_budget_disabled = False
        self.hsl_fact_capacity = 0
        self.hsl_factual_replay = False

    def _seed_initial_factual_capacity(self, batch_size):
        return  # V9 has no factual history allocation.

    def _prepare_replay_state_layout(self, library, library_args):
        if not self._temporal_replay_enabled():
            self._replay_state_bytes = None
            self._replay_states.clear()
            return
        if library_args not in self._replay_state_sizes:
            size = torch.empty(1, dtype=torch.int32, device=gpu_device())
            getattr(library, self.replay_state_size_kernel_name)(size, threads=1)
            self._replay_state_sizes[library_args] = int(size.item())
        state_bytes = self._replay_state_sizes[library_args]
        if self._replay_state_bytes != state_bytes:
            self._replay_states.clear()
        self._replay_state_bytes = state_bytes
        if self._replay_state_bytes < 1:
            raise RuntimeError("GPU replay state has an invalid compiled size")

    def _dispatch_replay(self, library, kernel_args, end_steps, *, batch_size, scalars):
        # Metal creates the compute pipeline lazily at kernel lookup. Keep its
        # first-use cost separate from device execution, including unchunked replay.
        profile_pipeline = getattr(self, "_profile_pipeline", False)
        kernel, pipeline_seconds, _prepared = _lookup_replay_kernel(
            library, self.replay_kernel_name, profile=profile_pipeline
        )
        if profile_pipeline:
            self._pipeline_seconds += pipeline_seconds
        if not self._temporal_replay_enabled():
            dispatch_options = {"threads": (batch_size, 1, 1)}
            dispatch_options.update(_metal_replay_group_options(kernel, batch_size))
            if gpu_device(torch) == "cuda" and self.replay_sides == 1 and self.cuda_whole_group_size is not None:
                # Spread independent, state-heavy candidates across more SMs.
                dispatch_options["group_size"] = self.cuda_whole_group_size
            kernel(
                *kernel_args, **dispatch_options
            )
            return
        adaptive = self.native_factual_hsl and self.bars.device.type == "cuda"
        history_limit = (
            CUDA_FACTUAL_HSL_REPLAY_CHUNK_BARS
            if adaptive and getattr(self, "hsl_fact_capacity", 0)
            else self.temporal_chunk_bar_limit
        )
        chunk_bars = min(
            history_limit,
            MPS_TM_MULTICOIN_CHUNK_CANDIDATE_STEPS // batch_size,
            self.max_dispatch_candidate_bars // (batch_size * self.n_coins * self.replay_sides),
        )
        if chunk_bars < 1:
            raise ValueError("MPS replay batch exceeds the per-dispatch work envelope")
        if self._history_batch_bytes(batch_size) > self.hsl_scratch_budget_bytes:
            raise ValueError("GPU replay state batch exceeds its scratch budget")
        state_key = (batch_size, self._replay_state_bytes)
        if state_key not in self._replay_states:
            self._replay_states.clear()
            self._replay_states = {
                state_key: torch.empty(
                    (batch_size, self._replay_state_bytes),
                    dtype=torch.uint8,
                    device=gpu_device(),
                )
            }
        replay_states = self._replay_states[state_key]
        stop_k = int(end_steps.max().item())
        dispatch_count = 0
        max_dispatch_seconds = 0.0
        controller = ReplayDurationController(chunk_bars) if adaptive else None
        adaptive_adjustments = 0
        min_processed_bars, max_processed_bars = chunk_bars, 0
        replay_progress = TemporalReplayProgress(batch_size, stop_k - 1, history_chunk_bars=chunk_bars)
        replay_started = time.perf_counter()
        next_progress = replay_started + 30.0
        # Serial candidates need independent CUDA blocks to spread across SMs.
        # Preserve the existing Apple SIMD-width scheduling.
        threads_per_threadgroup = 1 if self.bars.device.type == "cuda" else min(batch_size, 32)
        begin_k = 1
        while True:
            self.interrupt_check()
            dispatch_bars = controller.bars if controller is not None else chunk_bars
            replay_range = torch.tensor(
                [begin_k, min(begin_k + dispatch_bars, stop_k)],
                dtype=torch.int32,
                device=gpu_device(),
            )
            started = time.perf_counter()
            kernel(
                *kernel_args,
                replay_states,
                replay_range,
                threads=(batch_size, 1, 1),
                group_size=(threads_per_threadgroup, 1, 1),
            )
            # Bound queued work as well as each command, and make Ctrl+C visible
            # between temporal chunks even when profiling is disabled.
            synchronize()
            dispatch_count += 1
            dispatch_seconds = time.perf_counter() - started
            max_dispatch_seconds = max(max_dispatch_seconds, dispatch_seconds)
            completed_k = min(begin_k + dispatch_bars, stop_k)
            processed_bars = completed_k - begin_k
            min_processed_bars = min(min_processed_bars, processed_bars)
            max_processed_bars = max(max_processed_bars, processed_bars)
            record_replay_chunk(
                batch_size, processed_bars, stop_k - 1, dispatch_seconds,
                eligible=begin_k > 1 and processed_bars == dispatch_bars,
            )
            # Read compact producer statuses only after the completed chunk.
            # Fatal inputs/malformed facts still take precedence over capacity
            # growth across the entire batch; rejected work publishes no output.
            try:
                _require_available_held_valuation(scalars.detach().cpu())
            except HslFactHistoryOverflow:
                if not getattr(self, "_defer_hsl_overflow", False):
                    self._last_temporal_dispatch = {
                        "dispatch_count": dispatch_count,
                        "rejected_at_bar": completed_k - 1,
                        "threads_per_threadgroup": threads_per_threadgroup,
                    }
                    raise
            now = time.perf_counter()
            if now >= next_progress and completed_k < stop_k:
                replay_progress.log("progress", completed_k - 1, now - replay_started, kernel_dispatches=dispatch_count)
                next_progress = now + 30.0
            if completed_k >= stop_k:
                break
            if controller is not None:
                controller.observe(processed_bars, dispatch_seconds)
                adaptive_adjustments += int(controller.bars != dispatch_bars)
            begin_k = completed_k
        self.interrupt_check()
        replay_progress.log("complete", stop_k - 1, time.perf_counter() - replay_started, kernel_dispatches=dispatch_count)
        self._last_temporal_dispatch = {
            "dispatch_count": dispatch_count,
            "temporal_chunk_bars": chunk_bars,
            "threads_per_threadgroup": threads_per_threadgroup,
            "max_dispatch_seconds": max_dispatch_seconds,
            "kernel_candidate_steps": int((end_steps - 1).clamp(min=0).sum().item()),
            "replay_state_bytes_per_candidate": self._replay_state_bytes,
        }
        if adaptive:
            self._last_temporal_dispatch.update(
                adaptive_temporal_chunks=True,
                adaptive_chunk_adjustments=adaptive_adjustments,
                dispatch_duration_target_seconds=controller.target_seconds,
                temporal_chunk_bars_min=min_processed_bars,
                temporal_chunk_bars_max=max_processed_bars,
            )

    def _temporal_replay_enabled(self):
        return self.max_dispatch_candidate_bars is not None

    def run(self, params, *, profile=False, end_steps=None):
        """Run once with constant retained risk state; rejected work is fatal."""
        self._last_temporal_dispatch = None
        self._prepare_native_factual_hsl(params)
        self.last_hsl_fact_retries = 0
        return self._run_factual_attempt(params, profile=profile, end_steps=end_steps)

    def _run_factual_attempt(
        self,
        params: np.ndarray,
        *,
        profile: bool = False,
        end_steps: np.ndarray | None = None,
    ) -> dict:
        started = time.perf_counter() if profile else 0.0
        matrix = self._pack_params(params)
        bootstrap_counts = pack_bootstrap_counts(
            params, self.parameter_keys, self._bootstrap_coin_overrides,
            interval_minutes=self.interval_minutes, available=self.n)
        _validate_global_memory_params(params, self.parameter_keys, sides=self.replay_sides)
        validate_hourly_bootstrap_inputs(
            params, self.parameter_keys, self._bootstrap_coin_overrides,
            first_observed=self._hourly_first_observed, coin_ranges=self._hourly_coin_ranges,
            requested_start=self.requested_start_idx, warmup_bars=self.run_config.warmup_bars,
            available=self.n, end_steps=end_steps,
            side_activation=tuple(self.side_activation[side] for side in
                                  (("long", "short") if self.replay_sides == 2 else (self.side,))),
        )
        # Host feature proofs consume public policy columns; the shader matrix
        # additionally carries private decimal operands after each side block.
        policy_matrix = np.asarray(params, dtype=np.float32)
        self.dispatch_hsl_disabled = self._use_disabled_hsl_specialization(policy_matrix)
        self.dispatch_unstuck_ema_enabled = self._unstuck_ema_required(policy_matrix)
        self.dispatch_unstuck_enabled = self._unstuck_required(policy_matrix)
        packed = time.perf_counter() if profile else 0.0
        loader, library_args = self._library_cache_call()
        library, cold = _cached_library_with_miss(loader, *library_args)
        # The compiled ABI owns the state size. Include it in admission before
        # any candidate histories or persistent state buffers are allocated.
        self._prepare_replay_state_layout(library, library_args)
        if self._disable_native_cache_over_budget():
            loader, library_args = self._library_cache_call()
            library, scalar_cold = _cached_library_with_miss(loader, *library_args)
            cold |= scalar_cold
            self._prepare_replay_state_layout(library, library_args)
        compiled = time.perf_counter() if profile else 0.0
        if self._history_bytes_per_candidate():
            split = self._run_history_batches(
                params, profile=profile, end_steps=end_steps
            )
            if split is not None:
                if profile:
                    self.last_profile["cpu_pack_seconds"] += packed - started
                    self.last_profile["compile_seconds"] += compiled - packed
                    self.last_profile["cold"] |= cold
                    self.last_profile["cold_dispatch_count"] += int(cold)
                return split
        if self.hsl_capacity:
            keys = self.parameter_keys
            policy_matrix = np.asarray(params, dtype=np.float32)
            effective = (self._native_hsl_effective_enablement(policy_matrix, keys)
                         if getattr(self, "native_factual_hsl", True) else None)
            if policy_matrix.ndim == 2 and policy_matrix.shape[1] == len(keys):
                policy_matrix = np.concatenate((policy_matrix, policy_matrix), axis=1)
                if effective is not None:
                    effective = np.repeat(effective, 2, axis=1)
            self._validate_hsl_params(
                policy_matrix, keys, effective_enabled=effective)
        # The baseline split and policy validation precede any new headroom query.
        # Admission preserves that split width and the already-sized compiled ABI.
        self._seed_initial_factual_capacity(int(matrix.shape[0]))
        # Factual slots are fixed; fused side-policy ABI appends the short flag.
        factual_offset = 14 if self.replay_sides == 2 else 12
        self.settings[factual_offset] = getattr(self, "hsl_fact_capacity", 0)
        factual_replay = bool(getattr(self, "hsl_factual_replay", False))
        if factual_replay and not getattr(self, "hsl_fact_capacity", 0):
            raise ValueError("GPU factual HSL evaluation requires retained fill capacity")
        self.settings[factual_offset + 1] = float(factual_replay)
        batch_size = int(matrix.shape[0])
        self._prepare_scratch_shape(batch_size)
        params_mps = torch.as_tensor(matrix, device=gpu_device())
        bootstrap_counts_mps = torch.as_tensor(bootstrap_counts, device=gpu_device())
        end_steps_mps = self._end_steps(end_steps, batch_size)
        daily, scalars, gaps, coin_fill_counts = self._output_buffers(batch_size)
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
        hsl_ema_tail_samples = self._hsl_ema_tail_sample_buffer(batch_size)
        equity_balance_diff = self._equity_balance_diff_buffer(batch_size)
        entry_interval_stats, entry_interval_counts = self._entry_interval_buffers(
            batch_size
        )
        sizes_key = (batch_size, int(matrix.shape[1]))
        if sizes_key not in self._sizes:
            size_values = [
                batch_size,
                self.n,
                self.n_coins,
                self.n_days,
                self.requested_start_idx,
                self.run_config.warmup_bars,
                self.start_minute_of_day,
                self.start_minute_of_hour,
            ]
            if self.recovery_distribution_enabled:
                size_values.extend([self.recovery_stride, self.n_recovery_samples])
            self._sizes[sizes_key] = torch.tensor(
                size_values,
                dtype=torch.int32,
                device=gpu_device(),
            )
        prepared = time.perf_counter() if profile else 0.0
        self._profile_pipeline = profile
        self._pipeline_seconds = 0.0
        if profile:
            synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = prepared
        self._dispatch(
            library,
            params_mps,
            bootstrap_counts_mps,
            self._sizes[sizes_key],
            end_steps_mps,
            daily,
            scalars,
            gaps,
            coin_fill_counts,
            equity_balance_diff,
            entry_interval_stats,
            entry_interval_counts,
            recovery_samples,
            hsl_ema_tail_samples,
            batch_size=batch_size,
        )
        if profile:
            synchronize()
            finished = time.perf_counter()
            self.last_profile = {
                "cpu_pack_seconds": packed - started,
                "upload_and_zero_seconds": prepared - compiled,
                "compile_seconds": compiled - packed,
                "pre_dispatch_sync_seconds": dispatched - prepared,
                "pipeline_seconds": self._pipeline_seconds,
                "kernel_seconds": max(0.0, finished - dispatched - self._pipeline_seconds),
                "batch_size": batch_size,
                "dispatch_count": 1,
                "kernel_candidate_steps": int(
                    (end_steps_mps - 1).clamp(min=0).sum().item()
                ),
                "cold": cold,
            }
        else:
            self.last_profile = {}
            wait_for_cuda_stream()
        output = self._decode(daily, scalars, gaps)
        if hsl_ema_tail_samples is not None:
            # Decode rejects overflow/malformed attempts before any metrics are
            # reduced or published. Join only compact accepted sub-batch results.
            ema_tails = drawdown_ema_tail_from_samples(hsl_ema_tail_samples)
            for index, scope in enumerate(self._hsl_ema_tail_scopes()):
                output[f"hsl_drawdown_ema_mean_worst_1pct_{scope}"] = ema_tails[:, index]
        if self.raw_strategy_risk_enabled:
            output["raw_strategy_day_max_dd"] = daily[:, :, -1]
        if self.raw_strategy_growth_enabled:
            offset = self.daily_cols - int(self.raw_strategy_risk_enabled) - 2
            output["raw_strategy_day_end_eq"] = daily[:, :, offset]
            output["raw_strategy_day_min_eq"] = daily[:, :, offset + 1]
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        if self._exact_entry_intervals_enabled():
            output["entry_interval_native_metrics"] = entry_intervals_from_counts(
                entry_interval_counts, int(self.run_config.interval_ms),
            )
        else:
            output.update(
                _decode_entry_interval_outputs(entry_interval_stats, entry_interval_counts)
            )
        if self.recovery_distribution_enabled:
            sample_interval_days = self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            if self.compact_recovery_output:
                # Reduce accepted physical replays before logical sub-batches
                # clone/join outputs. Raw histories remain available to direct
                # diagnostic callers through the default output mode.
                output["strategy_eq_recovery_distribution"] = strategy_eq_recovery_distribution_from_samples(
                    recovery_samples, sample_interval_days=sample_interval_days,
                )
            else:
                output["strategy_eq_recovery_samples"] = recovery_samples
                output["strategy_eq_recovery_sample_interval_days"] = sample_interval_days
        if self.collect_coin_fill_counts:
            output["coin_fill_counts"] = coin_fill_counts
        if profile:
            synchronize()
            self.last_profile["metric_decode_seconds"] = time.perf_counter() - finished
        return output


class MpsEmaAnchorMulticoinRunner(_MulticoinReplayRunner):
    """EMA Anchor parameter layout, packing and kernel dispatch."""

    hsl_native_cache_enabled = False

    parameter_keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    coin_override_cols = EMA_ANCHOR_COIN_OVERRIDE_COLS
    coin_override_label = "EMA"
    coin_override_wel_column = EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
    coin_override_cooldown_column = EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN
    coin_override_hsl_start = EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN
    coin_override_unstuck_start = EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN

    def _prepare_coin_overrides(self, coin_overrides: np.ndarray) -> np.ndarray:
        return _scale_ema_multicoin_coin_overrides(
            coin_overrides, self.interval_minutes
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                f"expected multicoin EMA parameter matrix with {expected} columns, got {got}"
            )
        return np.ascontiguousarray(
            _scale_directional_minute_parameters(
                params,
                EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
                sides=1,
                interval_minutes=self.interval_minutes,
                ranking_coin_counts=getattr(self, "rms_ranking_coin_counts", None),
                cooldown_coin_overrides=getattr(self, "rms_cooldown_coin_overrides", None),
                dynamic_wel_by_tradability=self.dynamic_wel_by_tradability,
                side_activation=tuple(self.side_activation[side] for side in
                                      (("long", "short") if self.replay_sides == 2 else (self.side,))),
            ),
            dtype=np.float32,
        )

    replay_kernel_name = "passivbot_ema_anchor_multicoin"
    replay_state_size_kernel_name = "passivbot_ema_multicoin_replay_state_bytes"

    def _temporal_replay_enabled(self):
        return (self.max_dispatch_candidate_bars is not None
                and self.native_factual_hsl)

    def _dispatch(
        self,
        library,
        params_mps,
        bootstrap_counts_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        equity_balance_diff,
        entry_interval_stats,
        entry_interval_counts,
        recovery_samples,
        hsl_ema_tail_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.market_execution_ticks,
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.hour_log_ranges,
            self.coin_settings,
            *self.organillo_buffers,
            self.coin_overrides,
            params_mps,
            bootstrap_counts_mps,
            self.settings,
            _pack_replay_control(sizes, end_steps, self.recovery_distribution_enabled),
        )
        if self.btc_prices_enabled:
            kernel_args += (self.btc_prices,)
        if self.equity_balance_diff_enabled:
            kernel_args += (equity_balance_diff,)
        if self.entry_interval_enabled:
            kernel_args += (entry_interval_stats, entry_interval_counts)
        kernel_args += (
            daily,
            scalars,
            gaps,
            coin_fill_counts,
        )
        if self.recovery_distribution_enabled:
            kernel_args += (recovery_samples,)
        if self._hsl_ema_tail_samples_enabled():
            kernel_args += (hsl_ema_tail_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        self._dispatch_replay(library, kernel_args, end_steps, batch_size=batch_size, scalars=scalars)

    def _use_disabled_hsl_specialization(self, matrix):
        if getattr(self, "hsl_fact_capacity", 0):
            return False
        # Coin-mode forced delists still report per-coin panic segments when
        # HSL is disabled. The compact layout preserves only aggregate state.
        # Fused and TM kernels do not implement this one-side EMA layout.
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        return bool(
            getattr(self, "hsl_disabled_specialization", True)
            and not self.coin_hsl_may_enable
            and np.all(matrix[:, keys.index("hsl_enabled")] <= 0.5)
            and np.isin(matrix[:, keys.index("hsl_signal_mode")], [0, 1]).all()
        )

    def _library_cache_call(self):
        return _ema_anchor_multicoin_shader_library, (
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.hsl_raw_tail_enabled,
            self.recovery_distribution_enabled,
            self.dynamic_wel_by_tradability,
            self.btc_risk_enabled,
            self.equity_balance_diff_enabled,
            getattr(self, "dispatch_hsl_disabled", False),
            self.cuda_coin_capacity,
            self.hsl_capacity,
            self.pnl_lookback_bars,
            self.mps_coin_capacity,
            self.raw_strategy_risk_enabled,
            self.raw_strategy_growth_enabled,
            self.hsl_raw_tail_capacity,
            getattr(self, "dispatch_unstuck_ema_enabled", True),
            int(bool(getattr(self, "hsl_fact_capacity", 0))),
            getattr(self, "native_factual_hsl", True),
            getattr(self, "dispatch_unstuck_enabled", True),
            self._compiled_side(),
            self._temporal_replay_enabled(),
            self._hsl_native_cache_capacity(),
            getattr(self, "exposure_rms_enabled", False),
            self.organillo_enabled,
        )


class MpsEmaAnchorMulticoinFusedRunner(MpsEmaAnchorMulticoinRunner):
    """Persistent dual-side shared-account EMA Anchor runner on Apple MPS."""

    scalar_cols = MPS_MULTICOIN_FUSED_SCALAR_COLS
    replay_sides = 2
    replay_kernel_name = "passivbot_ema_anchor_multicoin_fused"
    replay_state_size_kernel_name = "passivbot_ema_multicoin_fused_replay_state_bytes"
    # The compact disabled-HSL layout specializes the one-side kernel. Keep
    # the fused dual-side kernel on its full state layout until every fused
    # HSL access has its own compile-time specialization.
    hsl_disabled_specialization = False

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        long_coin_overrides: np.ndarray | None = None,
        short_coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        filter_by_min_effective_cost: dict[str, bool],
        side_activation: dict[str, bool],
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        compact_recovery_output: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        hedge_mode: bool = True,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        max_dispatch_candidate_bars: int | None = None,
        factual_hsl: bool = True,
        exposure_rms_enabled: bool = False,
        interrupt_check=None,
        selection=None,
    ):
        super().__init__(
            run,
            data,
            side="long",
            coin_overrides=long_coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            filter_by_min_effective_cost=filter_by_min_effective_cost,
            side_activation=side_activation,
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market_long,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            hsl_raw_tail_enabled=hsl_raw_tail_enabled,
            recovery_distribution_enabled=recovery_distribution_enabled,
            compact_recovery_output=compact_recovery_output,
            raw_strategy_risk_enabled=raw_strategy_risk_enabled,
            raw_strategy_growth_enabled=raw_strategy_growth_enabled,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
            pnl_lookback_bars=pnl_lookback_bars,
            max_dispatch_candidate_bars=max_dispatch_candidate_bars,
            factual_hsl=factual_hsl,
            exposure_rms_enabled=exposure_rms_enabled,
            selection=selection,
            interrupt_check=interrupt_check,
        )
        if short_coin_overrides is None:
            short_coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols),
                np.nan,
                dtype=np.float32,
            )
        short_coin_overrides = np.asarray(short_coin_overrides, dtype=np.float64)
        expected_shape = (self.n_coins, self.coin_override_cols)
        if short_coin_overrides.shape != expected_shape:
            raise ValueError(
                "expected fused multicoin EMA short override matrix shaped "
                f"{expected_shape}, got {short_coin_overrides.shape}"
            )
        self._bootstrap_coin_overrides += (np.array(short_coin_overrides, dtype=np.float64, copy=True),)
        short_hsl_overrides = short_coin_overrides[
            :, EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN
        ]
        self.coin_hsl_may_enable = bool(
            self.coin_hsl_may_enable
            or np.any(np.isfinite(short_hsl_overrides) & (short_hsl_overrides > 0.5))
        )
        self._coin_hsl_enabled_overrides += (short_hsl_overrides.copy(),)
        self.rms_ranking_coin_counts += (int(np.count_nonzero(
            short_coin_overrides[:, EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN] != 0.0
        )),)
        self.rms_cooldown_coin_overrides += (short_coin_overrides[:, [
            EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
            self.coin_override_cols - 4, self.coin_override_cols - 3,
            self.coin_override_cols - 1, EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
        ]].copy(),)
        unstuck_start = EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN
        self._unstuck_ema_overrides += (short_coin_overrides[:, [unstuck_start, unstuck_start + 1]].copy(),)
        self.short_coin_overrides = torch.as_tensor(
            self._prepare_coin_overrides(short_coin_overrides), device=gpu_device()
        )
        max_realized_loss_pct = float(max_realized_loss_pct)
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        market_order_slippage_pct = float(market_order_slippage_pct)
        market_order_near_touch_threshold = float(market_order_near_touch_threshold)
        liq_floor = max(0.0, run.starting_balance) * max(0.0, run.liquidation_threshold)
        minimum_cost_filters = validate_minimum_cost_filters(filter_by_min_effective_cost)
        self.settings = torch.tensor(
            [
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                0.0,
                float(forager_score_hysteresis_pct),
                encoded_max_realized_loss_pct,
                float(self.collect_coin_fill_counts),
                market_order_slippage_pct,
                float(bool(hsl_panic_market_long)),
                float(bool(hsl_panic_market_short)),
                float(bool(hedge_mode)),
                float(bool(market_orders_allowed)),
                market_order_near_touch_threshold,
                float(minimum_cost_filters["long"]),
                0.0,  # Runtime factual-ring capacity; absent from compile identity.
                0.0,  # Internal factual evaluation control, independently of capture.
                float(self.loss_gate_enabled),  # Original float64 gate activation.
                float(minimum_cost_filters["short"]),  # Fused side-policy ABI 2.
                float(self.side_activation["long"]),  # Fused activation ABI 1.
                float(self.side_activation["short"]),
            ],
            dtype=torch.float32,
            device=gpu_device(),
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(EMA_ANCHOR_MULTICOIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected fused multicoin EMA parameter matrix with "
                f"{expected} columns, got {got}"
            )
        return np.ascontiguousarray(
            _scale_directional_minute_parameters(
                params,
                EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
                sides=2,
                interval_minutes=self.interval_minutes,
                ranking_coin_counts=getattr(self, "rms_ranking_coin_counts", None),
                cooldown_coin_overrides=getattr(self, "rms_cooldown_coin_overrides", None),
                dynamic_wel_by_tradability=self.dynamic_wel_by_tradability,
                side_activation=tuple(self.side_activation[side] for side in
                                      (("long", "short") if self.replay_sides == 2 else (self.side,))),
            ),
            dtype=np.float32,
        )

    def _dispatch(
        self,
        library,
        params_mps,
        bootstrap_counts_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        equity_balance_diff,
        entry_interval_stats,
        entry_interval_counts,
        recovery_samples,
        hsl_ema_tail_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.market_execution_ticks,
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.hour_log_ranges,
            self.coin_settings,
            *self.organillo_buffers,
            self.coin_overrides,
            self.short_coin_overrides,
            params_mps,
            bootstrap_counts_mps,
            self.settings,
            _pack_replay_control(sizes, end_steps, self.recovery_distribution_enabled),
        )
        if self.btc_prices_enabled:
            kernel_args += (self.btc_prices,)
        if self.equity_balance_diff_enabled:
            kernel_args += (equity_balance_diff,)
        if self.entry_interval_enabled:
            kernel_args += (entry_interval_stats, entry_interval_counts)
        kernel_args += (
            daily,
            scalars,
            gaps,
            coin_fill_counts,
        )
        if self.recovery_distribution_enabled:
            kernel_args += (recovery_samples,)
        if self._hsl_ema_tail_samples_enabled():
            kernel_args += (hsl_ema_tail_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        self._dispatch_replay(library, kernel_args, end_steps, batch_size=batch_size, scalars=scalars)

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled,
                             native_factual_hsl=getattr(self, "native_factual_hsl", True),
                             exposure_rms_enabled=getattr(self, "exposure_rms_enabled", False),
                             interval_ms=self.run_config.interval_ms)






class MpsTrailingMartingaleMulticoinRunner(_MulticoinReplayRunner):
    """Persistent single-side multi-coin Trailing Martingale proxy on MPS."""

    hsl_native_coin_cache_enabled = False

    replay_kernel_name = "passivbot_trailing_martingale_multicoin"
    replay_state_size_kernel_name = "passivbot_tm_multicoin_replay_state_bytes"
    replay_sides = 1
    cuda_whole_group_size = (32, 1, 1)

    parameter_keys = TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
    coin_override_cols = TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    coin_override_label = "Trailing Martingale"
    decimal_coin_overrides = True
    coin_override_wel_column = TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
    coin_override_cooldown_column = TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN
    coin_override_hsl_start = TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN
    coin_override_unstuck_start = TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN
    supports_entry_interval = True
    specialize_metal_coin_capacity = True

    def _prepare_coin_overrides(self, coin_overrides: np.ndarray) -> np.ndarray:
        scaled = _scale_tm_multicoin_coin_overrides(coin_overrides, self.interval_minutes)
        keys = [key for key, _ in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS]
        columns = [keys.index(key) for key in _TM_DECIMAL_KEYS]
        metadata = _tm_decimal_metadata(coin_overrides[:, columns]).reshape(len(scaled), -1)
        flags = _tm_modifier_masks(coin_overrides, keys, overrides=True)
        for key in ("entry_retracement_base_pct", "close_retracement_base_pct"):
            column = keys.index(key)
            underflow = (coin_overrides[:, column] > 0.0) & (scaled[:, column] == 0.0)
            scaled[underflow, column] = np.finfo(np.float32).tiny
        return np.ascontiguousarray(np.concatenate((scaled, metadata, flags), axis=1), dtype=np.float32)

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        side: str,
        coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        filter_by_min_effective_cost: dict[str, bool],
        side_activation: dict[str, bool],
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        compact_recovery_output: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        max_dispatch_candidate_bars: int | None = None,
        factual_hsl: bool = True,
        exposure_rms_enabled: bool = False,
        interrupt_check=None,
        selection=None,
    ):
        self.loss_gate_enabled = float(max_realized_loss_pct) < 1.0
        self.loss_gate_specialization = True
        super().__init__(
            run,
            data,
            side=side,
            coin_overrides=coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            filter_by_min_effective_cost=filter_by_min_effective_cost,
            side_activation=side_activation,
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            hsl_raw_tail_enabled=hsl_raw_tail_enabled,
            recovery_distribution_enabled=recovery_distribution_enabled,
            compact_recovery_output=compact_recovery_output,
            raw_strategy_risk_enabled=raw_strategy_risk_enabled,
            raw_strategy_growth_enabled=raw_strategy_growth_enabled,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
            pnl_lookback_bars=pnl_lookback_bars,
            max_dispatch_candidate_bars=max_dispatch_candidate_bars,
            factual_hsl=factual_hsl,
            exposure_rms_enabled=exposure_rms_enabled,
            selection=selection,
            interrupt_check=interrupt_check,
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS)
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected multicoin Trailing Martingale parameter matrix with "
                f"{expected} columns, got {got}"
            )
        scaled = _scale_directional_minute_parameters(
            params,
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
            sides=1,
            interval_minutes=self.interval_minutes,
            ranking_coin_counts=getattr(self, "rms_ranking_coin_counts", None),
            cooldown_coin_overrides=getattr(self, "rms_cooldown_coin_overrides", None),
            dynamic_wel_by_tradability=self.dynamic_wel_by_tradability,
                side_activation=tuple(self.side_activation[side] for side in
                                      (("long", "short") if self.replay_sides == 2 else (self.side,))),
        )
        return _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, sides=1
        )

    def _library_cache_call(self):
        args = (
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.hsl_raw_tail_enabled,
            self.recovery_distribution_enabled,
            self.dynamic_wel_by_tradability,
            self.btc_risk_enabled,
            self.equity_balance_diff_enabled,
            self.entry_interval_enabled,
            self.max_dispatch_candidate_bars is not None,
            self.cuda_coin_capacity,
            self.hsl_capacity,
            self.pnl_lookback_bars,
            self.mps_coin_capacity,
            self.loss_gate_specialization and not self.loss_gate_enabled,
            self.raw_strategy_risk_enabled,
            self.raw_strategy_growth_enabled,
            self.hsl_raw_tail_capacity,
            getattr(self, "dispatch_unstuck_ema_enabled", True),
            int(bool(getattr(self, "hsl_fact_capacity", 0))),
            getattr(self, "native_factual_hsl", True),
            getattr(self, "dispatch_unstuck_enabled", True),
            self._compiled_side(),
            self._hsl_native_cache_capacity(),
            getattr(self, "exposure_rms_enabled", False),
            getattr(self, "coin_overrides_absent", False),
            bool(self._hsl_native_cache_capacity() and self._hsl_native_coin_cache_selected()),
            self.organillo_enabled,
        )
        return _trailing_martingale_multicoin_shader_library, args

    def _dispatch(
        self,
        library,
        params_mps,
        bootstrap_counts_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        equity_balance_diff,
        entry_interval_stats,
        entry_interval_counts,
        recovery_samples,
        hsl_ema_tail_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.market_execution_ticks,
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.touch_nearest_ticks,
            self.touch_min_qty_bits,
            self.touch_min_qty_relation,
            self.hour_log_ranges,
            self.coin_settings,
            *self.organillo_buffers,
            self.coin_overrides,
            params_mps,
            bootstrap_counts_mps,
            self.settings,
            _pack_replay_control(sizes, end_steps, self.recovery_distribution_enabled),
        )
        if self.btc_prices_enabled:
            kernel_args += (self.btc_prices,)
        if self.equity_balance_diff_enabled:
            kernel_args += (equity_balance_diff,)
        if self.entry_interval_enabled:
            kernel_args += (entry_interval_stats, entry_interval_counts)
        kernel_args += (
            daily,
            scalars,
            gaps,
            coin_fill_counts,
        )
        if self.recovery_distribution_enabled:
            kernel_args += (recovery_samples,)
        if self._hsl_ema_tail_samples_enabled():
            kernel_args += (hsl_ema_tail_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        self._dispatch_replay(library, kernel_args, end_steps, batch_size=batch_size, scalars=scalars)



class MpsTrailingMartingaleMulticoinFusedRunner(MpsTrailingMartingaleMulticoinRunner):
    """Persistent dual-side shared-account Trailing Martingale runner on MPS."""

    replay_kernel_name = "passivbot_trailing_martingale_multicoin_fused"
    replay_state_size_kernel_name = "passivbot_tm_multicoin_fused_replay_state_bytes"
    replay_sides = 2

    scalar_cols = MPS_MULTICOIN_FUSED_SCALAR_COLS

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        long_coin_overrides: np.ndarray | None = None,
        short_coin_overrides: np.ndarray | None = None,
        forager_score_hysteresis_pct: float = 0.0,
        max_realized_loss_pct: float = 1.0,
        collect_coin_fill_counts: bool = False,
        filter_by_min_effective_cost: dict[str, bool],
        side_activation: dict[str, bool],
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        compact_recovery_output: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        hedge_mode: bool = True,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        max_dispatch_candidate_bars: int | None = None,
        factual_hsl: bool = True,
        exposure_rms_enabled: bool = False,
        interrupt_check=None,
        selection=None,
    ):
        super().__init__(
            run,
            data,
            side="long",
            coin_overrides=long_coin_overrides,
            forager_score_hysteresis_pct=forager_score_hysteresis_pct,
            max_realized_loss_pct=max_realized_loss_pct,
            collect_coin_fill_counts=collect_coin_fill_counts,
            filter_by_min_effective_cost=filter_by_min_effective_cost,
            side_activation=side_activation,
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market_long,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            hsl_raw_tail_enabled=hsl_raw_tail_enabled,
            recovery_distribution_enabled=recovery_distribution_enabled,
            compact_recovery_output=compact_recovery_output,
            raw_strategy_risk_enabled=raw_strategy_risk_enabled,
            raw_strategy_growth_enabled=raw_strategy_growth_enabled,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
            pnl_lookback_bars=pnl_lookback_bars,
            max_dispatch_candidate_bars=max_dispatch_candidate_bars,
            factual_hsl=factual_hsl,
            exposure_rms_enabled=exposure_rms_enabled,
            selection=selection,
            interrupt_check=interrupt_check,
        )
        if short_coin_overrides is None:
            short_coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols),
                np.nan,
                dtype=np.float32,
            )
        short_coin_overrides = np.asarray(short_coin_overrides, dtype=np.float64)
        expected_shape = (self.n_coins, self.coin_override_cols)
        if short_coin_overrides.shape != expected_shape:
            raise ValueError(
                "expected fused multicoin Trailing Martingale short override "
                f"matrix shaped {expected_shape}, got {short_coin_overrides.shape}"
            )
        self._bootstrap_coin_overrides += (np.array(short_coin_overrides, dtype=np.float64, copy=True),)
        short_hsl_overrides = short_coin_overrides[:, TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN]
        self.coin_hsl_may_enable = bool(self.coin_hsl_may_enable or np.any(
            np.isfinite(short_hsl_overrides) & (short_hsl_overrides > 0.5)))
        self._coin_hsl_enabled_overrides += (short_hsl_overrides.copy(),)
        self.rms_ranking_coin_counts += (int(np.count_nonzero(
            short_coin_overrides[:, TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN] != 0.0
        )),)
        self.rms_cooldown_coin_overrides += (short_coin_overrides[:, [
            TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
            self.coin_override_cols - 4, self.coin_override_cols - 3,
            self.coin_override_cols - 1, TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
        ]].copy(),)
        unstuck_start = TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN
        self._unstuck_ema_overrides += (short_coin_overrides[:, [unstuck_start, unstuck_start + 1]].copy(),)
        prepared_overrides = self._prepare_coin_overrides(short_coin_overrides)
        self.coin_overrides_absent = (
            self.coin_overrides_absent and not bool(np.isfinite(short_coin_overrides).any())
        )
        self.short_coin_overrides = torch.as_tensor(prepared_overrides, device=gpu_device())
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            float(max_realized_loss_pct)
        )
        liq_floor = max(0.0, run.starting_balance) * max(0.0, run.liquidation_threshold)
        minimum_cost_filters = validate_minimum_cost_filters(filter_by_min_effective_cost)
        self.settings = torch.tensor(
            [
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                0.0,
                float(forager_score_hysteresis_pct),
                encoded_max_realized_loss_pct,
                float(self.collect_coin_fill_counts),
                float(market_order_slippage_pct),
                float(bool(hsl_panic_market_long)),
                float(bool(hsl_panic_market_short)),
                float(bool(hedge_mode)),
                float(bool(market_orders_allowed)),
                float(market_order_near_touch_threshold),
                float(minimum_cost_filters["long"]),
                0.0,  # Runtime factual-ring capacity; absent from compile identity.
                0.0,  # Internal factual evaluation control, independently of capture.
                float(self.loss_gate_enabled),  # Original float64 gate activation.
                float(minimum_cost_filters["short"]),  # Fused side-policy ABI 2.
                float(self.side_activation["long"]),  # Fused activation ABI 1.
                float(self.side_activation["short"]),
            ],
            dtype=torch.float32,
            device=gpu_device(),
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        expected = len(TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected fused multicoin Trailing Martingale parameter matrix "
                f"with {expected} columns, got {got}"
            )
        scaled = _scale_directional_minute_parameters(
            params,
            TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
            sides=2,
            interval_minutes=self.interval_minutes,
            ranking_coin_counts=getattr(self, "rms_ranking_coin_counts", None),
            cooldown_coin_overrides=getattr(self, "rms_cooldown_coin_overrides", None),
            dynamic_wel_by_tradability=self.dynamic_wel_by_tradability,
                side_activation=tuple(self.side_activation[side] for side in
                                      (("long", "short") if self.replay_sides == 2 else (self.side,))),
        )
        return _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, sides=2
        )

    def _dispatch(
        self,
        library,
        params_mps,
        bootstrap_counts_mps,
        sizes,
        end_steps,
        daily,
        scalars,
        gaps,
        coin_fill_counts,
        equity_balance_diff,
        entry_interval_stats,
        entry_interval_counts,
        recovery_samples,
        hsl_ema_tail_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.market_execution_ticks,
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.touch_nearest_ticks,
            self.touch_min_qty_bits,
            self.touch_min_qty_relation,
            self.hour_log_ranges,
            self.coin_settings,
            *self.organillo_buffers,
            self.coin_overrides,
            self.short_coin_overrides,
            params_mps,
            bootstrap_counts_mps,
            self.settings,
            _pack_replay_control(sizes, end_steps, self.recovery_distribution_enabled),
        )
        if self.btc_prices_enabled:
            kernel_args += (self.btc_prices,)
        if self.equity_balance_diff_enabled:
            kernel_args += (equity_balance_diff,)
        if self.entry_interval_enabled:
            kernel_args += (entry_interval_stats, entry_interval_counts)
        kernel_args += (
            daily,
            scalars,
            gaps,
            coin_fill_counts,
        )
        if self.recovery_distribution_enabled:
            kernel_args += (recovery_samples,)
        if self._hsl_ema_tail_samples_enabled():
            kernel_args += (hsl_ema_tail_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        self._dispatch_replay(library, kernel_args, end_steps, batch_size=batch_size, scalars=scalars)

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled,
                             native_factual_hsl=getattr(self, "native_factual_hsl", True),
                             exposure_rms_enabled=getattr(self, "exposure_rms_enabled", False),
                             interval_ms=self.run_config.interval_ms)
