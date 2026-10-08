from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
import torch

from optimization.gpu.metric_registry import (
    WEIGHTED_EQUITY_METRICS, WEIGHTED_RAW_EQUITY_METRICS,
    WEIGHTED_ACCOUNT_EQUITY_METRICS,
)
from optimization.gpu.weighted_equity import (
    weighted_equity_from_samples, weighted_equity_history_bytes,
)

from optimization.gpu.specialization import unstuck_ema_required
from optimization.gpu.replay_progress import TemporalReplayProgress
from optimization.gpu.autotune import record_replay_chunk
from optimization.gpu.runtime import (
    gpu_device,
    compile_shader,
    synchronize,
    wait_for_cuda_stream,
)

from optimization.gpu.model import (
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
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    GAP_BINS,
    MPS_MULTICOIN_MAX_COINS,
    MPS_TM_MULTICOIN_CHUNK_BARS,
    MPS_TM_SINGLE_COIN_CHUNK_BARS,
    MPS_TM_MULTICOIN_CHUNK_CANDIDATE_STEPS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
    TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS,
    TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS,
    TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
    encode_tm_retracement_base_pct,
    single_coin_shader_topology,
)

MPS_DAILY_COLS = 8
MPS_MULTICOIN_DAILY_COLS = 9
MPS_SCALAR_COLS = 32
MPS_MULTICOIN_BASE_SCALAR_COLS = 62
MPS_MULTICOIN_EMA_TAIL_SCALAR_COLS = 64
MPS_MULTICOIN_RAW_DRAWDOWN_SCALAR_COLS = 66
MPS_MULTICOIN_SCALAR_COLS = 68
MPS_DIRECTIONAL_BASE_SCALAR_COLS = 66
MPS_DIRECTIONAL_EMA_TAIL_SCALAR_COLS = 68
MPS_DIRECTIONAL_RAW_DRAWDOWN_SCALAR_COLS = 70
MPS_DIRECTIONAL_SCALAR_COLS = 72
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
_TM_TRAILING_ENTRY_ONLY_DEFINE = "#define PASSIVBOT_TM_TRAILING_ENTRY_ONLY 1\n"
_TM_RECURSIVE_ENTRY_ONLY_DEFINE = "#define PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY 1\n"
_TM_TRAILING_CLOSE_ONLY_DEFINE = "#define PASSIVBOT_TM_TRAILING_CLOSE_ONLY 1\n"
_TM_REDUCERS_DISABLED_DEFINE = "#define PASSIVBOT_TM_REDUCERS_DISABLED 1\n"
_TM_MARKET_ORDERS_DISABLED_DEFINE = "#define PASSIVBOT_TM_MARKET_ORDERS_DISABLED 1\n"
_TM_LOSS_GATE_DISABLED_DEFINE = "#define PASSIVBOT_TM_LOSS_GATE_DISABLED 1\n"
_TM_VOLATILITY_DISABLED_DEFINE = "#define PASSIVBOT_TM_VOLATILITY_DISABLED 1\n"


def _hsl_layout(capacity: int, fact_capacity: int = 0) -> tuple[int, int]:
    blocks = (capacity + 63) // 64
    tree_size = 1 << (blocks - 1).bit_length()
    if type(fact_capacity) is not int or fact_capacity < 0:
        raise ValueError("Invalid GPU HSL factual capacity")
    storage_nodes = 2 * tree_size + (capacity + 3) // 4
    if fact_capacity:
        storage_nodes += 2 + fact_capacity + (fact_capacity + 1) // 2
    return tree_size, storage_nodes


def _with_hsl(source: str, capacity: int, fact_capacity: int = 0) -> str:
    if fact_capacity and not capacity:
        raise ValueError("GPU HSL facts require an HSL window")
    if not capacity:
        return source
    if not 1 <= capacity <= 90 * 1440 + 2:
        raise ValueError("Invalid GPU HSL window capacity")
    tree_size, _ = _hsl_layout(capacity, fact_capacity)
    return (
        f"#define PASSIVBOT_HSL_FACTS_ENABLED {int(bool(fact_capacity))}\n"
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


def _with_tm_dispatch_features(
    source: str,
    *,
    trailing_entry_only: bool,
    recursive_entry_only: bool,
    trailing_close_only: bool,
    reducers_disabled: bool,
    market_orders_disabled: bool,
    loss_gate_disabled: bool,
    volatility_disabled: bool,
) -> str:
    if trailing_entry_only and recursive_entry_only:
        raise ValueError(
            "TM trailing-entry-only and recursive-entry-only modes are mutually exclusive"
        )
    features = (
        (
            trailing_entry_only,
            "#ifndef PASSIVBOT_TM_TRAILING_ENTRY_ONLY",
            _TM_TRAILING_ENTRY_ONLY_DEFINE,
        ),
        (
            recursive_entry_only,
            "#ifndef PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY",
            _TM_RECURSIVE_ENTRY_ONLY_DEFINE,
        ),
        (
            trailing_close_only,
            "#ifndef PASSIVBOT_TM_TRAILING_CLOSE_ONLY",
            _TM_TRAILING_CLOSE_ONLY_DEFINE,
        ),
        (
            reducers_disabled,
            "#ifndef PASSIVBOT_TM_REDUCERS_DISABLED",
            _TM_REDUCERS_DISABLED_DEFINE,
        ),
        (
            market_orders_disabled,
            "#ifndef PASSIVBOT_TM_MARKET_ORDERS_DISABLED",
            _TM_MARKET_ORDERS_DISABLED_DEFINE,
        ),
        (
            loss_gate_disabled,
            "#ifndef PASSIVBOT_TM_LOSS_GATE_DISABLED",
            _TM_LOSS_GATE_DISABLED_DEFINE,
        ),
        (
            volatility_disabled,
            "#ifndef PASSIVBOT_TM_VOLATILITY_DISABLED",
            _TM_VOLATILITY_DISABLED_DEFINE,
        ),
    )
    for enabled, marker, define in features:
        if not enabled:
            continue
        if marker not in source:
            raise RuntimeError(
                f"MPS source is missing the TM dispatch feature guard {marker}"
            )
        source = define + source
    return source


def _encode_max_realized_loss_pct(value: float) -> float:
    """Encode a float64 loss fraction without loosening its Metal budget."""

    if value >= 1.0:
        return 1.0
    encoded = np.float32(value)
    if float(encoded) > value:
        encoded = np.nextafter(encoded, np.float32(-np.inf))
    return float(encoded)


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


def _pack_tm_parameter_matrix(
    params: np.ndarray, keys: tuple[str, ...], *, sides: int
) -> np.ndarray:
    """Pack TM rows while preserving each candidate's retracement mode sign."""

    matrix = np.ascontiguousarray(params, dtype=np.float32)
    side_width = len(keys)
    for side_index in range(sides):
        offset = side_index * side_width
        for key in (
            "entry_retracement_base_pct",
            "close_retracement_base_pct",
        ):
            column = offset + keys.index(key)
            positive_underflow = (params[:, column] > 0.0) & (
                matrix[:, column] == np.float32(0.0)
            )
            if np.any(positive_underflow):
                matrix[positive_underflow, column] = encode_tm_retracement_base_pct(
                    np.finfo(np.float64).tiny
                )
    return matrix


def _tm_dispatch_specialization(
    matrix: np.ndarray,
    *,
    long_enabled: bool,
    short_enabled: bool,
    market_orders_allowed: bool,
    loss_gate_enabled: bool,
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    """Prove dispatch-wide TM features before compiling away inactive paths."""

    side_width = len(TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS)
    active_offsets = [
        side_index * side_width
        for side_index, enabled in enumerate((long_enabled, short_enabled))
        if enabled
    ]

    def all_active_rows(predicate, key: str) -> bool:
        key_index = TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS.index(key)
        return bool(
            active_offsets
            and matrix.shape[0] > 0
            and all(
                predicate(matrix[:, offset + key_index]) for offset in active_offsets
            )
        )

    trailing_entry_only = all_active_rows(
        lambda values: np.all(np.isfinite(values) & (values > 0.0)),
        "entry_retracement_base_pct",
    )
    recursive_entry_only = all_active_rows(
        lambda values: np.all(np.isfinite(values) & (values <= 0.0)),
        "entry_retracement_base_pct",
    )
    trailing_close_only = all_active_rows(
        lambda values: np.all(np.isfinite(values) & (values > 0.0)),
        "close_retracement_base_pct",
    )
    reducers_disabled = all(
        all_active_rows(
            lambda values: np.all(np.isfinite(values) & (values <= 0.5)),
            key,
        )
        for key in (
            "wel_enforcer_enabled",
            "twel_enforcer_enabled",
            "unstuck_enabled",
        )
    )
    volatility_disabled = all(
        all_active_rows(
            lambda values: np.all(np.isfinite(values) & (values == 0.0)),
            key,
        )
        for key in (
            "entry_threshold_volatility_1h_weight",
            "entry_threshold_volatility_1m_weight",
            "entry_retracement_volatility_1h_weight",
            "entry_retracement_volatility_1m_weight",
            "close_threshold_volatility_1h_weight",
            "close_threshold_volatility_1m_weight",
            "close_retracement_volatility_1h_weight",
            "close_retracement_volatility_1m_weight",
        )
    )
    return (
        trailing_entry_only,
        recursive_entry_only,
        trailing_close_only,
        reducers_disabled,
        not market_orders_allowed,
        not loss_gate_enabled,
        volatility_disabled,
    )


def _upgrade_legacy_single_coin_wel_params(
    params: np.ndarray, *, side_width: int
) -> np.ndarray:
    """Upgrade pre-independent-EMA rows, with or without the legacy WEL column.

    The old ABI used the strategy horizons for unstuck. Preserve those horizons
    when accepting its rows; current producers always supply explicit spans.
    """
    if params.ndim != 2 or params.shape[1] == side_width * 2:
        return params
    legacy_width = params.shape[1] // 2
    previous_width = side_width - len(ADAPTIVE_PARAM_KEYS)
    if params.shape[1] % 2 or legacy_width not in (
        previous_width,
        previous_width - 2,
        previous_width - 3,
    ):
        return params
    strategy_start = 1 if side_width == len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS) else 0
    sides = []
    defaults = np.array([0.0, -1.0, 0.0, 0.0, 60.0, 0.0, 1200.0], dtype=params.dtype)
    for offset in (0, legacy_width):
        side = params[:, offset : offset + legacy_width]
        parts = [side]
        if legacy_width == previous_width - 3:
            parts.append(np.full((len(params), 1), -1.0, dtype=params.dtype))
        if legacy_width != previous_width:
            parts.append(side[:, strategy_start : strategy_start + 2])
        parts.append(np.broadcast_to(defaults, (len(params), len(defaults))))
        sides.append(np.concatenate(parts, axis=1))
    return np.concatenate(sides, axis=1)


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
        "hsl_cooldown_minutes_after_red",
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
            hsl_span_column = offset + keys.index("hsl_ema_span_minutes")
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


def _scale_single_coin_minute_parameters(
    params: np.ndarray,
    keys: tuple[str, ...],
    *,
    sides: int,
    interval_minutes: float,
) -> np.ndarray:
    """Compatibility wrapper for the original single-coin helper name."""

    return _scale_directional_minute_parameters(
        params,
        keys,
        sides=sides,
        interval_minutes=interval_minutes,
    )


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
        hsl_start_column + 3,
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
    # Shared replay reserves the portfolio tail plus two timing moments;
    # retained directional single-coin replay reserves only the moments.
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


def _cached_library_with_miss(loader, *args):
    misses_before = loader.cache_info().misses
    library = loader(*args)
    return library, loader.cache_info().misses > misses_before


@lru_cache(maxsize=16)
def _shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    hsl_capacity: int = 0,
    hsl_raw_tail_capacity: int = 1,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_source_py(),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return compile_shader(_with_hsl(source, hsl_capacity))


@lru_cache(maxsize=4)
def _ema_anchor_long_no_hsl_shader_library(
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_recovery_distribution(
        passivbot_rust.mps_ema_anchor_long_no_hsl_source_py(),
        recovery_distribution_enabled,
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return compile_shader(source)


@lru_cache(maxsize=4)
def _ema_anchor_short_no_hsl_shader_library(
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_recovery_distribution(
        passivbot_rust.mps_ema_anchor_short_no_hsl_source_py(),
        recovery_distribution_enabled,
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return compile_shader(source)


@lru_cache(maxsize=16)
def _trailing_martingale_shader_library(
    trailing_entry_only: bool = False,
    recursive_entry_only: bool = False,
    trailing_close_only: bool = False,
    reducers_disabled: bool = False,
    market_orders_disabled: bool = False,
    loss_gate_disabled: bool = False,
    volatility_disabled: bool = False,
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    entry_interval_enabled: bool = False,
    hsl_diagnostics_enabled: bool = True,
    temporal_chunking: bool = False,
    hsl_capacity: int = 0,
    hsl_raw_tail_capacity: int = 1,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_tm_dispatch_features(
        passivbot_rust.mps_trailing_martingale_source_py(),
        trailing_entry_only=trailing_entry_only,
        recursive_entry_only=recursive_entry_only,
        trailing_close_only=trailing_close_only,
        reducers_disabled=reducers_disabled,
        market_orders_disabled=market_orders_disabled,
        loss_gate_disabled=loss_gate_disabled,
        volatility_disabled=volatility_disabled,
    )
    source = _with_hsl_features(
        source,
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
        diagnostics_enabled=hsl_diagnostics_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    if temporal_chunking:
        source = "#define PASSIVBOT_TM_SINGLE_COIN_TEMPORAL_REPLAY 1\n" + source
    return compile_shader(_with_hsl(source, hsl_capacity))


@lru_cache(maxsize=16)
def _trailing_martingale_long_hsl_shader_library(
    trailing_entry_only: bool = False,
    recursive_entry_only: bool = False,
    trailing_close_only: bool = False,
    reducers_disabled: bool = False,
    market_orders_disabled: bool = False,
    loss_gate_disabled: bool = False,
    volatility_disabled: bool = False,
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    entry_interval_enabled: bool = False,
    hsl_diagnostics_enabled: bool = True,
    hsl_raw_tail_capacity: int = 1,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_tm_dispatch_features(
        passivbot_rust.mps_trailing_martingale_long_hsl_source_py(),
        trailing_entry_only=trailing_entry_only,
        recursive_entry_only=recursive_entry_only,
        trailing_close_only=trailing_close_only,
        reducers_disabled=reducers_disabled,
        market_orders_disabled=market_orders_disabled,
        loss_gate_disabled=loss_gate_disabled,
        volatility_disabled=volatility_disabled,
    )
    source = _with_hsl_features(
        source,
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
        diagnostics_enabled=hsl_diagnostics_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return compile_shader(source)


@lru_cache(maxsize=16)
def _trailing_martingale_short_hsl_shader_library(
    trailing_entry_only: bool = False,
    recursive_entry_only: bool = False,
    trailing_close_only: bool = False,
    reducers_disabled: bool = False,
    market_orders_disabled: bool = False,
    loss_gate_disabled: bool = False,
    volatility_disabled: bool = False,
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    entry_interval_enabled: bool = False,
    hsl_diagnostics_enabled: bool = True,
    hsl_raw_tail_capacity: int = 1,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_tm_dispatch_features(
        passivbot_rust.mps_trailing_martingale_short_hsl_source_py(),
        trailing_entry_only=trailing_entry_only,
        recursive_entry_only=recursive_entry_only,
        trailing_close_only=trailing_close_only,
        reducers_disabled=reducers_disabled,
        market_orders_disabled=market_orders_disabled,
        loss_gate_disabled=loss_gate_disabled,
        volatility_disabled=volatility_disabled,
    )
    source = _with_hsl_features(
        source,
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
        diagnostics_enabled=hsl_diagnostics_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return compile_shader(source)


@lru_cache(maxsize=8)
def _trailing_martingale_long_no_hsl_shader_library(
    trailing_entry_only: bool = False,
    recursive_entry_only: bool = False,
    trailing_close_only: bool = False,
    reducers_disabled: bool = False,
    market_orders_disabled: bool = False,
    loss_gate_disabled: bool = False,
    volatility_disabled: bool = False,
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    entry_interval_enabled: bool = False,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_tm_dispatch_features(
        passivbot_rust.mps_trailing_martingale_long_no_hsl_source_py(),
        trailing_entry_only=trailing_entry_only,
        recursive_entry_only=recursive_entry_only,
        trailing_close_only=trailing_close_only,
        reducers_disabled=reducers_disabled,
        market_orders_disabled=market_orders_disabled,
        loss_gate_disabled=loss_gate_disabled,
        volatility_disabled=volatility_disabled,
    )
    source = _with_recovery_distribution(
        source,
        recovery_distribution_enabled,
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return compile_shader(source)


@lru_cache(maxsize=8)
def _trailing_martingale_short_no_hsl_shader_library(
    trailing_entry_only: bool = False,
    recursive_entry_only: bool = False,
    trailing_close_only: bool = False,
    reducers_disabled: bool = False,
    market_orders_disabled: bool = False,
    loss_gate_disabled: bool = False,
    volatility_disabled: bool = False,
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
    entry_interval_enabled: bool = False,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_tm_dispatch_features(
        passivbot_rust.mps_trailing_martingale_short_no_hsl_source_py(),
        trailing_entry_only=trailing_entry_only,
        recursive_entry_only=recursive_entry_only,
        trailing_close_only=trailing_close_only,
        reducers_disabled=reducers_disabled,
        market_orders_disabled=market_orders_disabled,
        loss_gate_disabled=loss_gate_disabled,
        volatility_disabled=volatility_disabled,
    )
    source = _with_recovery_distribution(
        source,
        recovery_distribution_enabled,
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return compile_shader(source)


def _with_unstuck_ema(source: str, enabled: bool) -> str:
    if enabled:
        return source
    if "#if PASSIVBOT_UNSTUCK_EMA_ENABLED" not in source:
        raise RuntimeError("GPU source is missing the unstuck EMA ablation contract")
    return "#define PASSIVBOT_UNSTUCK_EMA_ENABLED 0\n" + source


def _with_unstuck_pnl_window(source, lookback_bars, capacity):
    if lookback_bars:
        source = (
            f"#define PASSIVBOT_UNSTUCK_PNL_LOOKBACK_BARS {lookback_bars}\n"
            f"#define PASSIVBOT_UNSTUCK_PNL_CAPACITY {capacity}\n" + source
        )
    return source


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
    unstuck_pnl_lookback_bars: int = 0,
    unstuck_pnl_capacity: int = 0,
    weighted_volume_enabled: bool = False,
    raw_strategy_risk_enabled: bool = False,
    raw_strategy_growth_enabled: bool = False,
    weighted_raw_equity_enabled: bool = False,
    weighted_account_equity_enabled: bool = False,
    hsl_raw_tail_capacity: int = 1,
    unstuck_ema_enabled: bool = True,
    hsl_fact_capacity: int = 0,
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
    source = _with_unstuck_ema(source, unstuck_ema_enabled)
    # Keep diagnostics for forced delist panic-loss parity even when the HSL
    # controllers and their per-candle scans are compiled away.
    source = _with_hsl_disabled(source, hsl_disabled)
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    if weighted_raw_equity_enabled:
        source = "#define PASSIVBOT_WEIGHTED_RAW_EQUITY_ENABLED 1\n" + source
    if weighted_account_equity_enabled:
        source = "#define PASSIVBOT_WEIGHTED_ACCOUNT_EQUITY_ENABLED 1\n" + source
    if raw_strategy_risk_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_RISK_ENABLED 1\n" + source
    if raw_strategy_growth_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_GROWTH_ENABLED 1\n" + source
    if weighted_volume_enabled:
        source = "#define PASSIVBOT_WEIGHTED_VOLUME_ENABLED 1\n" + source
    source = _with_dynamic_wel_by_tradability(source, dynamic_wel_by_tradability)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    if hsl_capacity:
        source = f"#define PASSIVBOT_HSL_LOOKBACK {hsl_lookback}\n" + _with_hsl(
            source, hsl_capacity, hsl_fact_capacity
        )
    source = _with_unstuck_pnl_window(source, unstuck_pnl_lookback_bars, unstuck_pnl_capacity)
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
    unstuck_pnl_lookback_bars: int = 0,
    unstuck_pnl_capacity: int = 0,
    loss_gate_disabled: bool = False,
    weighted_volume_enabled: bool = False,
    raw_strategy_risk_enabled: bool = False,
    raw_strategy_growth_enabled: bool = False,
    weighted_raw_equity_enabled: bool = False,
    weighted_account_equity_enabled: bool = False,
    hsl_raw_tail_capacity: int = 1,
    unstuck_ema_enabled: bool = True,
    hsl_fact_capacity: int = 0,
):
    gpu_device(torch)
    import passivbot_rust

    source = _with_hsl_features(
        _with_tm_dispatch_features(
            passivbot_rust.mps_trailing_martingale_multicoin_source_py(),
            trailing_entry_only=False,
            recursive_entry_only=False,
            trailing_close_only=False,
            reducers_disabled=False,
            market_orders_disabled=False,
            volatility_disabled=False,
            loss_gate_disabled=loss_gate_disabled,
        ),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
        raw_tail_capacity=hsl_raw_tail_capacity,
    )
    source = _with_unstuck_ema(source, unstuck_ema_enabled)
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    if weighted_raw_equity_enabled:
        source = "#define PASSIVBOT_WEIGHTED_RAW_EQUITY_ENABLED 1\n" + source
    if weighted_account_equity_enabled:
        source = "#define PASSIVBOT_WEIGHTED_ACCOUNT_EQUITY_ENABLED 1\n" + source
    if raw_strategy_risk_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_RISK_ENABLED 1\n" + source
    if raw_strategy_growth_enabled:
        source = "#define PASSIVBOT_RAW_STRATEGY_GROWTH_ENABLED 1\n" + source
    if weighted_volume_enabled:
        source = "#define PASSIVBOT_WEIGHTED_VOLUME_ENABLED 1\n" + source
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
    source = _with_unstuck_pnl_window(source, unstuck_pnl_lookback_bars, unstuck_pnl_capacity)
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


@lru_cache(maxsize=1)
def _weighted_volume_shader_library():
    import passivbot_rust

    return compile_shader(passivbot_rust.mps_weighted_volume_source_py())


def _volume_history_bytes(sample_capacity: int):
    # Float2 samples, a possible contiguous copy, float2 bounds, scalar result
    # and four size integers (conservatively charged to every candidate).
    return int(sample_capacity) * 16 + 28


def weighted_volume_from_samples(
    samples, first_eq_ts, last_eq_ts, *, start_minute_of_day, interval_minutes,
):
    """Reduce canonical fill-day averages over actual equity-horizon suffixes."""
    if samples.device.type not in {"mps", "cuda"} or samples.dtype != torch.float32:
        raise ValueError("weighted volume requires float32 GPU samples")
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError("weighted volume expects batch-by-step-by-two samples")
    batch_size, capacity, _ = samples.shape
    for bounds in (first_eq_ts, last_eq_ts):
        if (bounds.device != samples.device or bounds.shape != (batch_size,)
                or bounds.dtype != torch.float32):
            raise ValueError("weighted volume requires matching GPU equity bounds")
    interval_minutes = int(interval_minutes)
    if interval_minutes < 1 or not 0 <= int(start_minute_of_day) < 1440:
        raise ValueError("weighted volume requires a positive minute interval and UTC origin")
    matrix = samples.contiguous()
    output = torch.empty(batch_size, dtype=torch.float32, device=matrix.device)
    if batch_size == 0:
        return output
    bounds = torch.stack((first_eq_ts, last_eq_ts), dim=1)
    sizes = torch.tensor(
        [batch_size, capacity, int(start_minute_of_day), interval_minutes],
        dtype=torch.int32, device=matrix.device,
    )
    _weighted_volume_shader_library().passivbot_weighted_volume(
        matrix, bounds, output, sizes, threads=(batch_size, 1, 1),
    )
    return output


def _recovery_history_bytes(sample_capacity: int):
    # Samples, a possible contiguous copy of a truncated view, the index stack
    # and duration histogram. Output and scaled result coexist until completion.
    return int(sample_capacity) * 16 + MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS * 8


class HslFactHistoryOverflow(RuntimeError):
    """A rejected GPU attempt may be repeated with larger worker-owned storage."""


def _require_available_held_valuation(scalars):
    # Scalar 9 normally holds -1 (not liquidated) or a liquidation day >= 0.
    # Metal writes -2 and returns immediately if a held coin has no price.
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


def _decode_outputs(daily, scalars, gaps, *, btc_risk_enabled=False) -> dict:
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
        # Shared replay keeps the packed ABI labels but stores elapsed steps;
        # retained directional single-coin replay still stores sample counts.
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
    if btc_risk_enabled:
        output.update(_decode_btc_risk_outputs(daily, active_days, 9))
    return output


def _decode_multicoin_fused_outputs(daily, scalars, gaps, *, btc_risk_enabled=False) -> dict:
    output = _decode_outputs(daily, scalars, gaps, btc_risk_enabled=btc_risk_enabled)
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


def _decode_directional_outputs(daily, scalars, gaps) -> dict:
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
        "day_net_pnl": daily[:, :, 5],
        "day_last_fill_balance": daily[:, :, 6],
        "day_fill_count": daily[:, :, 7],
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
        "short_psize": scalars[:, 15],
        "short_pprice": scalars[:, 16],
        "hsl_long_enabled": scalars[:, 18] > 0.0,
        "hsl_short_enabled": scalars[:, 19] > 0.0,
        "hsl_triggers_long": scalars[:, 20],
        "hsl_triggers_short": scalars[:, 21],
        "hsl_restarts_long": scalars[:, 22],
        "hsl_restarts_short": scalars[:, 23],
        "hsl_tier_samples_total": scalars[:, 24],
        "hsl_tier_samples_red": scalars[:, 25],
        "hsl_duration_sum_steps": scalars[:, 26],
        "hsl_duration_max_steps": scalars[:, 27],
        "hsl_duration_count": scalars[:, 28],
        "hsl_trigger_drawdown_sum": scalars[:, 29],
        "hsl_trigger_drawdown_count": scalars[:, 30],
        "hsl_flatten_time_sum_steps": scalars[:, 31],
        "hsl_flatten_time_count": scalars[:, 32],
        "hsl_restart_retrigger_count": scalars[:, 33],
        "hsl_halt_to_restart_equity_loss": scalars[:, 34],
        "hsl_panic_close_loss_sum": scalars[:, 35],
        "hsl_panic_close_loss_max": scalars[:, 36],
        "hsl_panic_loss_drawdown_min": scalars[:, 37],
        "hsl_panic_loss_drawdown_sum": scalars[:, 38],
        "hsl_panic_loss_drawdown_max": scalars[:, 39],
        "hsl_panic_loss_drawdown_count": scalars[:, 40],
        "profit_sum": scalars[:, 41],
        "loss_sum": scalars[:, 42],
        "position_unchanged_max_ms": scalars[:, 43],
        "entry_initial_balance_pct_long": scalars[:, 44],
        "entry_initial_balance_pct_short": scalars[:, 45],
        "total_wallet_exposure_max": scalars[:, 46],
        "total_wallet_exposure_mean": scalars[:, 47],
        "fill_count": scalars[:, 48],
        "fill_count_entry": scalars[:, 49],
        "fill_count_long": scalars[:, 50],
        "fills_active_days_count": scalars[:, 51],
        "pnl_recovery_max_ms": scalars[:, 52],
        "held_sum_ms": scalars[:, 53],
        "held_sum_squared_hours": scalars[:, -2],
        "gap_sum_squared_hours": scalars[:, -1],
        "held_count": scalars[:, 54],
        "account_recovery_max_ms": scalars[:, 55],
        "profit_sum_long": scalars[:, 56],
        "loss_sum_long": scalars[:, 57],
        "profit_sum_short": scalars[:, 58],
        "loss_sum_short": scalars[:, 59],
        "hsl_drawdown_ema_max_long": scalars[:, 60],
        "hsl_drawdown_ema_max_short": scalars[:, 61],
        "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 62],
        "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 63],
        "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(scalars, 64),
        "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(scalars, 65),
        "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 66),
        "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 67),
        "hsl_drawdown_raw_mean_worst_1pct_long": _scalar_column_or_zero(scalars, 68),
        "hsl_drawdown_raw_mean_worst_1pct_short": _scalar_column_or_zero(scalars, 69),
    }
    output.update(_decode_btc_risk_outputs(daily, active_days, 8))
    return output


class MpsEmaAnchorRunner:
    hsl_scratch_budget_bytes = 512 * 1024 * 1024

    """Persistent single-coin Metal runner with invariant data resident on MPS."""

    def __init__(
        self,
        market: ProxyMarket,
        run: ProxyRun,
        data: dict,
        *,
        long_enabled: bool = True,
        short_enabled: bool = False,
        hedge_mode: bool = True,
        filter_by_min_effective_cost: bool = False,
        max_realized_loss_pct: float = 1.0,
        taker_fee: float | None = None,
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_enabled: bool = True,
        pnl_lookback_bars: int = 0,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
    ):
        self.hsl_capacity = 0
        self._hsl_scratch_buffers = {}
        self.market = market
        if entry_interval_enabled:
            raise ValueError(
                "MPS entry-interval output is only defined for Trailing Martingale"
            )
        self.run_config = run
        self.interval_minutes = float(run.interval_ms) / 60_000.0
        if (
            not np.isfinite(self.interval_minutes)
            or self.interval_minutes < 1.0
            or not self.interval_minutes.is_integer()
        ):
            raise ValueError(
                "MPS single-coin runner requires an integer candle interval "
                "of at least one minute"
            )
        self.long_enabled = bool(long_enabled)
        self.short_enabled = bool(short_enabled)
        self.hedge_mode = bool(hedge_mode)
        if not self.long_enabled and not self.short_enabled:
            raise ValueError("MPS EMA proxy requires at least one enabled side")
        max_realized_loss_pct = float(max_realized_loss_pct)
        if not np.isfinite(max_realized_loss_pct) or max_realized_loss_pct < 0.0:
            raise ValueError("max_realized_loss_pct must be finite and non-negative")
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        self.loss_gate_enabled = encoded_max_realized_loss_pct < 1.0
        self.market_orders_allowed = bool(market_orders_allowed)
        taker_fee = market.maker_fee if taker_fee is None else float(taker_fee)
        market_order_slippage_pct = float(market_order_slippage_pct)
        market_order_near_touch_threshold = float(market_order_near_touch_threshold)
        pnl_lookback_bars = int(pnl_lookback_bars)
        if not np.isfinite(taker_fee):
            raise ValueError("taker_fee must be finite")
        if (
            not np.isfinite(market_order_slippage_pct)
            or market_order_slippage_pct < 0.0
        ):
            raise ValueError(
                "market_order_slippage_pct must be finite and non-negative"
            )
        if (
            not np.isfinite(market_order_near_touch_threshold)
            or market_order_near_touch_threshold < 0.0
        ):
            raise ValueError(
                "market_order_near_touch_threshold must be finite and non-negative"
            )
        if pnl_lookback_bars < 0:
            raise ValueError("pnl_lookback_bars must be non-negative")
        self.pnl_lookback_bars = pnl_lookback_bars
        self.hsl_ema_tail_enabled = bool(hsl_ema_tail_enabled)
        self.hsl_raw_drawdown_enabled = bool(hsl_raw_drawdown_enabled)
        self.hsl_raw_tail_enabled = bool(hsl_raw_tail_enabled)
        self.shader_topology = single_coin_shader_topology(
            long_enabled=self.long_enabled,
            short_enabled=self.short_enabled,
            hsl_enabled=bool(hsl_enabled),
        )
        if self.shader_topology != "generic":
            self.hsl_ema_tail_enabled = False
            self.hsl_raw_drawdown_enabled = False
            self.hsl_raw_tail_enabled = False
        self.recovery_distribution_enabled = bool(recovery_distribution_enabled)
        self.n = int(data["n"])
        if bool(hsl_enabled) and (
            self.interval_minutes != 1 or not 1440 <= pnl_lookback_bars <= 90 * 1440
        ):
            raise ValueError("GPU HSL requires 1m candles and 1..90d lookback")
        self.hsl_capacity = min(self.n + 2, pnl_lookback_bars + 2)
        self.shader_topology = "generic"
        self.n_days = int(data["n_days"])
        self.hsl_raw_tail_capacity = (
            _raw_drawdown_tail_capacity(self.n_days) if self.hsl_raw_tail_enabled else 1
        )
        self.btc_prices = _btc_risk_price_tensor(btc_prices, expected_count=self.n)
        self.equity_balance_diff_enabled = bool(equity_balance_diff_enabled)
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
        self.daily_cols = MPS_DAILY_COLS + (3 if self.btc_risk_enabled else 0)
        self.recovery_stride = 1 if self.recovery_distribution_enabled else 0
        self.n_recovery_samples = (
            max(
                1,
                (self.n + self.recovery_stride - 1) // self.recovery_stride + 1,
            )
            if self.recovery_distribution_enabled
            else 1
        )
        self.bars = (
            torch.stack(
                [
                    data["high_f"],
                    data["low_f"],
                    data["close_f"],
                    data["log_range"],
                    data["hour_log_range"],
                ],
                dim=1,
            )
            .to(dtype=torch.float32, device=gpu_device())
            .contiguous()
        )
        self.flags = (
            torch.stack(
                [
                    data["valid"].to(torch.int32),
                    data["can_gen"].to(torch.int32),
                    data["day_idx"].to(torch.int32),
                    data["hour_valid"].to(torch.int32),
                    data["high_fill_max_tick"].to(torch.int32),
                    data["low_nonfill_max_tick"].to(torch.int32),
                    data["touch_down_tick"].to(torch.int32),
                    data["touch_up_tick"].to(torch.int32),
                    data["touch_nearest_tick"].to(torch.int32),
                    data["touch_min_qty_bits"].to(torch.int32),
                    data["touch_min_qty_relation"].to(torch.int32),
                ],
                dim=1,
            )
            .to(device=gpu_device())
            .contiguous()
        )
        liq_floor = max(0.0, run.starting_balance) * max(0.0, run.liquidation_threshold)
        self.settings = torch.tensor(
            [
                market.qty_step,
                market.price_step,
                market.min_qty,
                market.min_cost,
                market.c_mult,
                market.maker_fee,
                run.starting_balance,
                liq_floor,
                run.interval_ms,
                float(self.long_enabled),
                float(self.short_enabled),
                float(self.hedge_mode),
                float(bool(filter_by_min_effective_cost)),
                data["max_effective_min_cost"],
                encoded_max_realized_loss_pct,
                taker_fee,
                market_order_slippage_pct,
                float(bool(hsl_panic_market_long)),
                float(bool(hsl_panic_market_short)),
                float(bool(market_orders_allowed)),
                market_order_near_touch_threshold,
            ],
            dtype=torch.float32,
            device=gpu_device(),
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._recovery_buffers: dict[int, torch.Tensor] = {}
        self._equity_balance_diff_buffers: dict[int, torch.Tensor] = {}
        self._sizes: dict[tuple[int, int, int], torch.Tensor] = {}
        self.last_profile: dict[str, float | int | bool] = {}

    def _shader_library_cache_call(self):
        if self.shader_topology == "long_no_hsl":
            return _ema_anchor_long_no_hsl_shader_library, (
                self.recovery_distribution_enabled,
                self.btc_risk_enabled,
                self.equity_balance_diff_enabled,
            )
        if self.shader_topology == "short_no_hsl":
            return _ema_anchor_short_no_hsl_shader_library, (
                self.recovery_distribution_enabled,
                self.btc_risk_enabled,
                self.equity_balance_diff_enabled,
            )
        return _shader_library, (
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.hsl_raw_tail_enabled,
            self.recovery_distribution_enabled,
            self.btc_risk_enabled,
            self.equity_balance_diff_enabled,
            self.hsl_capacity,
            self.hsl_raw_tail_capacity,
        )

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        params = _upgrade_legacy_single_coin_wel_params(
            params, side_width=len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
        )
        expected = len(EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                f"expected directional EMA parameter matrix with {expected} columns, got {got}"
            )
        scaled = _scale_single_coin_minute_parameters(
            params,
            EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
            sides=2,
            interval_minutes=self.interval_minutes,
        )
        packed = np.ascontiguousarray(scaled, dtype=np.float32)
        self._validate_hsl_params(packed, EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS)
        return packed

    def _output_buffers(self, batch_size: int):
        if batch_size not in self._buffers:
            # Optimizer generations use one fixed batch size.  Keep only the
            # active allocation so benchmark/tuning calls with several sizes
            # do not retain every large daily-output buffer.
            self._buffers = {
                batch_size: (
                    torch.zeros(
                        (batch_size, self.n_days, self.daily_cols),
                        dtype=torch.float32,
                        device=gpu_device(),
                    ),
                    torch.zeros(
                        (
                            batch_size,
                            (
                                MPS_DIRECTIONAL_SCALAR_COLS
                                if self.hsl_raw_tail_enabled
                                else (
                                    MPS_DIRECTIONAL_RAW_DRAWDOWN_SCALAR_COLS
                                    if self.hsl_raw_drawdown_enabled
                                    else (
                                        MPS_DIRECTIONAL_EMA_TAIL_SCALAR_COLS
                                        if self.hsl_ema_tail_enabled
                                        else MPS_DIRECTIONAL_BASE_SCALAR_COLS
                                    )
                                )
                            ),
                        ),
                        dtype=torch.float32,
                        device=gpu_device(),
                    ),
                    torch.zeros(
                        (batch_size, GAP_BINS), dtype=torch.int32, device=gpu_device()
                    ),
                )
            }
        else:
            for buffer in self._buffers[batch_size]:
                buffer.zero_()
        # An untouched day has no valid equity sample. The kernel overwrites
        # this sentinel whenever it flushes an active day.
        self._buffers[batch_size][0][:, :, 1].fill_(float("inf"))
        return self._buffers[batch_size]

    def _validate_hsl_params(self, params, keys, *, effective_enabled=None):
        width = len(keys)
        for side, active in enumerate((self.long_enabled, self.short_enabled)):
            if not active:
                continue
            cols = {
                key: params[:, side * width + keys.index(key)]
                for key in (
                    "hsl_enabled",
                    "hsl_red_threshold",
                    "hsl_ema_span_minutes",
                    "hsl_cooldown_minutes_after_red",
                    "hsl_restart_policy",
                    "hsl_signal_mode",
                    "hsl_slot_count",
                )
            }
            enabled = cols["hsl_enabled"] > 0.5
            if not np.isfinite(cols["hsl_enabled"]).all():
                raise ValueError("GPU HSL enablement must be finite")
            if not np.any(enabled):
                continue
            needs_history = (enabled if effective_enabled is None
                             else enabled & effective_enabled[:, side])
            if np.any(needs_history) and (
                self.interval_minutes != 1
                or not 1440 <= self.pnl_lookback_bars <= 90 * 1440
            ):
                raise ValueError(
                    "Enabled GPU HSL requires 1m candles and 1..90d lookback"
                )
            if any(not np.isfinite(v[enabled]).all() for v in cols.values()):
                raise ValueError("GPU HSL parameters must be finite")

            def selected(key):
                return cols[key][enabled]

            if (
                np.any(selected("hsl_red_threshold") <= 0)
                or np.any(selected("hsl_red_threshold") > 1)
                or np.any(selected("hsl_ema_span_minutes") < 1)
                or np.any(selected("hsl_cooldown_minutes_after_red") < 0)
                or not np.isin(selected("hsl_restart_policy"), [0, 2]).all()
                or not np.isin(selected("hsl_signal_mode"), [0, 1, 2]).all()
                or np.any(selected("hsl_slot_count") < 1)
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
            "hsl_ema_span_minutes",
            "hsl_cooldown_minutes_after_red",
            "hsl_restart_policy",
        ):
            i = keys.index(key)
            if np.any(params[unified, i] != params[unified, width + i]):
                raise ValueError("Unified GPU HSL requires one shared policy")

    def release_replay_scratch(self):
        """Release completed replay buffers, retaining invariant market tensors."""
        for name in (
            "_hsl_scratch_buffers",
            "_buffers",
            "_recovery_buffers",
            "_equity_balance_diff_buffers",
            "_entry_interval_stat_buffers",
            "_entry_interval_count_buffers",
            "_replay_states",
            "_replay_state_sizes",
        ):
            buffers = getattr(self, name, None)
            if buffers is not None:
                buffers.clear()

    def _hsl_bytes_per_candidate(self):
        tree_size, storage_nodes = _hsl_layout(self.hsl_capacity)
        return 2 * (storage_nodes * 32 + self.hsl_capacity * 8)

    def _history_bytes_per_candidate(self):
        return (
            (self._hsl_bytes_per_candidate() if self.hsl_capacity else 0)
            + (_recovery_history_bytes(self.n_recovery_samples)
               if self.recovery_distribution_enabled else 0)
        )

    def _run_hsl_batches(self, params, **kwargs):
        """Partition independent candidates before allocating bounded history scratch."""
        history_bytes = self._history_bytes_per_candidate()
        if not history_bytes or params.ndim != 2:
            return None
        limit = self.hsl_scratch_budget_bytes // history_bytes
        if limit < 1:
            raise ValueError("GPU history exceeds the scratch budget")
        if len(params) <= limit:
            return None
        outputs, profiles = [], []
        for start in range(0, len(params), limit):
            result = self.run(params[start : start + limit], **kwargs)
            # Each dispatch reuses its output tensors. Retain only decoded
            # results, never all history scratch allocations.
            outputs.append(
                {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in result.items()
                }
            )
            profiles.append(dict(self.last_profile))
        combined = {}
        for key, value in outputs[0].items():
            if isinstance(value, torch.Tensor):
                combined[key] = torch.cat([o[key] for o in outputs], dim=0)
            else:
                if any(o[key] != value for o in outputs):
                    raise ValueError(f"GPU sub-batches disagree on {key}")
                combined[key] = value
        self.last_profile = {}
        if kwargs.get("profile"):
            self.last_profile = {
                key: sum(p.get(key, 0) for p in profiles)
                for key in profiles[0]
                if key.endswith("_seconds")
            }
            self.last_profile.update(
                batch_size=len(params),
                candidate_batch_count=len(outputs),
                dispatch_count=sum(p.get("dispatch_count", 1) for p in profiles),
                cold=any(p.get("cold", False) for p in profiles),
                effective_candle_count=profiles[0]["effective_candle_count"],
            )
        return combined

    def _hsl_buffers(self, batch_size):
        if not self.hsl_capacity:
            return ()
        tree_size, storage_nodes = _hsl_layout(self.hsl_capacity)
        nbytes = batch_size * self._hsl_bytes_per_candidate()
        if nbytes > self.hsl_scratch_budget_bytes:
            raise ValueError("HSL GPU batch exceeds its configured scratch budget")
        if batch_size not in self._hsl_scratch_buffers:
            self._hsl_scratch_buffers = {
                batch_size: (
                    torch.empty(
                        (batch_size, 2, storage_nodes, 32),
                        dtype=torch.uint8,
                        device=gpu_device(),
                    ),
                    torch.empty(
                        (batch_size, 2, 2 * self.hsl_capacity),
                        dtype=torch.int32,
                        device=gpu_device(),
                    ),
                )
            }
        return self._hsl_scratch_buffers[batch_size]

    def _recovery_sample_buffer(self, batch_size: int):
        if batch_size * self._history_bytes_per_candidate() > self.hsl_scratch_budget_bytes:
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

    def _single_coin_size_values(
        self,
        batch_size: int,
        parameter_count: int,
        *,
        end_step: int | None = None,
    ) -> list[int]:
        effective_end_step = self.n if end_step is None else int(end_step)
        if not 3 <= effective_end_step <= self.n:
            raise ValueError(
                "single-coin MPS end_step must be between 3 and the full candle "
                f"count {self.n}, got {effective_end_step}"
            )
        values = [
            int(batch_size),
            effective_end_step,
            self.n_days,
            int(parameter_count),
            self.run_config.first_valid_idx,
            self.pnl_lookback_bars,
            self.run_config.last_valid_idx,
        ]
        if self.recovery_distribution_enabled:
            values.extend([self.recovery_stride, self.n_recovery_samples])
        return values

    def run(
        self,
        params: np.ndarray,
        *,
        profile: bool = False,
        end_step: int | None = None,
    ) -> dict:
        batched = self._run_hsl_batches(params, profile=profile, end_step=end_step)
        if batched is not None:
            return batched
        started = time.perf_counter() if profile else 0.0
        matrix = self._pack_params(params)
        packed = time.perf_counter() if profile else 0.0
        params_mps = torch.as_tensor(matrix, device=gpu_device())
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
        equity_balance_diff = self._equity_balance_diff_buffer(batch_size)
        effective_end_step = self.n if end_step is None else int(end_step)
        sizes_key = (batch_size, int(matrix.shape[1]), effective_end_step)
        if sizes_key not in self._sizes:
            size_values = self._single_coin_size_values(
                batch_size,
                int(matrix.shape[1]),
                end_step=effective_end_step,
            )
            self._sizes[sizes_key] = torch.tensor(
                size_values,
                dtype=torch.int32,
                device=gpu_device(),
            )
        prepared = time.perf_counter() if profile else 0.0
        loader, library_args = self._shader_library_cache_call()
        library, cold = _cached_library_with_miss(loader, *library_args)
        compiled = time.perf_counter() if profile else 0.0
        if profile:
            synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled

        def dispatch_once():
            kernel_args = (
                self.bars,
                self.flags,
                params_mps,
                self.settings,
                self._sizes[sizes_key],
            )
            if self.btc_prices_enabled:
                kernel_args += (self.btc_prices,)
            if self.equity_balance_diff_enabled:
                kernel_args += (equity_balance_diff,)
            kernel_args += (
                daily,
                scalars,
                gaps,
            )
            kernel_args += self._hsl_buffers(batch_size)
            if self.recovery_distribution_enabled:
                kernel_args += (recovery_samples,)
            library.passivbot_ema_anchor(
                *kernel_args,
                threads=(batch_size, 1, 1),
            )

        dispatch_once()
        if profile:
            synchronize()
            finished = time.perf_counter()
            self.last_profile = {
                "cpu_pack_seconds": packed - started,
                "upload_and_zero_seconds": prepared - packed,
                "compile_seconds": compiled - prepared,
                "pre_dispatch_sync_seconds": dispatched - compiled,
                "kernel_seconds": finished - dispatched,
                "batch_size": batch_size,
                "dispatch_count": 1,
                "cold": cold,
                "effective_candle_count": effective_end_step,
            }
        else:
            self.last_profile = {}
            wait_for_cuda_stream()
        output = _decode_directional_outputs(daily, scalars, gaps)
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        if profile:
            synchronize()
            self.last_profile["metric_decode_seconds"] = time.perf_counter() - finished
        return output


class MpsEmaAnchorMulticoinRunner:
    """Persistent single-side multi-coin EMA Anchor screening runner on MPS."""

    coin_override_cols = EMA_ANCHOR_COIN_OVERRIDE_COLS
    coin_override_label = "EMA"
    scalar_cols = MPS_MULTICOIN_SCALAR_COLS

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
        filter_by_min_effective_cost: bool = False,
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        weighted_volume_enabled: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        weighted_equity_metrics=(),
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        unstuck_pnl_lookback_bars: int = 0,
        factual_hsl: bool = False,
        interrupt_check=None,
    ):
        if side not in {"long", "short"}:
            raise ValueError(
                f"MPS multicoin runner side must be long or short, got {side!r}"
            )
        self.side = side
        self.native_factual_hsl = bool(factual_hsl)
        self.interrupt_check = interrupt_check or (lambda: None)
        self.collect_coin_fill_counts = bool(collect_coin_fill_counts)
        self.hsl_ema_tail_enabled = bool(hsl_ema_tail_enabled)
        self.hsl_raw_drawdown_enabled = bool(hsl_raw_drawdown_enabled)
        self.hsl_raw_tail_enabled = bool(hsl_raw_tail_enabled)
        self.recovery_distribution_enabled = bool(recovery_distribution_enabled)
        self.weighted_volume_enabled = bool(weighted_volume_enabled)
        self.raw_strategy_risk_enabled = bool(raw_strategy_risk_enabled)
        self.raw_strategy_growth_enabled = bool(raw_strategy_growth_enabled)
        self.weighted_equity_metrics = frozenset(weighted_equity_metrics)
        if self.weighted_equity_metrics - WEIGHTED_EQUITY_METRICS:
            raise ValueError("unsupported resident weighted equity metrics")
        if self.weighted_equity_metrics and gpu_device(torch) != "cuda":
            raise ValueError("resident weighted equity capture currently requires CUDA")
        self.weighted_raw_equity_enabled = bool(
            self.weighted_equity_metrics & WEIGHTED_RAW_EQUITY_METRICS
        )
        self.weighted_account_equity_enabled = bool(
            self.weighted_equity_metrics & WEIGHTED_ACCOUNT_EQUITY_METRICS
        )
        self.weighted_equity_cols = int(self.weighted_raw_equity_enabled) + int(
            self.weighted_account_equity_enabled
        )
        self.dynamic_wel_by_tradability = bool(dynamic_wel_by_tradability)
        fused = self.scalar_cols == MPS_MULTICOIN_FUSED_SCALAR_COLS
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
        if unstuck_pnl_lookback_bars < 0:
            raise ValueError("unstuck_pnl_lookback_bars must be nonnegative")
        self.unstuck_pnl_lookback_bars = int(unstuck_pnl_lookback_bars)
        # At most one event per candle, including both lookback endpoints.
        # Coalescing preserves every intrabar peak; no fill-count cap is needed.
        self.unstuck_pnl_capacity = (
            min(self.n, self.unstuck_pnl_lookback_bars + 1)
            if self.unstuck_pnl_lookback_bars
            else 0
        )
        self._unstuck_pnl_buffers = {}

        self.pnl_lookback_bars = int(pnl_lookback_bars)
        self.hsl_capacity = 0
        self._hsl_scratch_buffers = {}
        self.hsl_scratch_budget_bytes = 512 * 1024 * 1024
        self.hsl_replay_sides = 2 if fused else 1
        self.hsl_scopes = self.hsl_replay_sides * (self.n_coins + 1)
        # Conservative resident seed; completed overflow learns upward. Keep
        # that estimate across dispatches where the effective HSL policy is off.
        self.hsl_fact_capacity_learned = min(256, max(1, self.n))
        self.hsl_fact_capacity = self.hsl_fact_capacity_learned if factual_hsl else 0
        self.hsl_factual_replay = bool(factual_hsl)
        if pnl_lookback_bars != 0 and (
            self.interval_minutes != 1 or not 1440 <= pnl_lookback_bars <= 90 * 1440
        ):
            raise ValueError("GPU HSL requires 1m candles and 1..90d lookback")
        self.hsl_capacity = min(self.n + 2, pnl_lookback_bars + 2)

        self.btc_prices = _btc_risk_price_tensor(btc_prices, expected_count=self.n)
        self.equity_balance_diff_enabled = bool(equity_balance_diff_enabled)
        self.entry_interval_enabled = bool(entry_interval_enabled)
        if (
            self.entry_interval_enabled
            and self.coin_override_label != "Trailing Martingale"
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
            and self.coin_override_label == "Trailing Martingale"
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
        if coin_overrides is None:
            coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols), np.nan, dtype=np.float32
            )
        coin_overrides = np.asarray(coin_overrides, dtype=np.float32)
        if coin_overrides.shape != (self.n_coins, self.coin_override_cols):
            raise ValueError(
                f"expected multicoin {self.coin_override_label} override matrix shaped "
                f"({self.n_coins}, {self.coin_override_cols}), "
                f"got {coin_overrides.shape}"
            )
        wel_column = (TRAILING_MARTINGALE_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN
                      if self.coin_override_label == "Trailing Martingale"
                      else EMA_ANCHOR_COIN_OVERRIDE_WALLET_EXPOSURE_COLUMN)
        self.rms_ranking_coin_counts = (int(np.count_nonzero(coin_overrides[:, wel_column] != 0.0)),)
        cooldown_column = (TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN
                           if self.coin_override_label == "Trailing Martingale"
                           else EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN)
        self.rms_cooldown_coin_overrides = (coin_overrides[:, [
            cooldown_column, self.coin_override_cols - 4, self.coin_override_cols - 3,
            self.coin_override_cols - 1, wel_column,
        ]].copy(),)
        hsl_start = (TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN
                     if self.coin_override_label == "Trailing Martingale"
                     else EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN)
        hsl_overrides = coin_overrides[:, hsl_start]
        self._coin_hsl_enabled_overrides = (hsl_overrides.copy(),)
        self.coin_hsl_may_enable = bool(
            np.any(np.isfinite(hsl_overrides) & (hsl_overrides > 0.5))
        )
        unstuck_start = (TRAILING_MARTINGALE_COIN_OVERRIDE_UNSTUCK_START_COLUMN
                         if self.coin_override_label == "Trailing Martingale"
                         else EMA_ANCHOR_COIN_OVERRIDE_UNSTUCK_START_COLUMN)
        self._unstuck_ema_overrides = (coin_overrides[:, [unstuck_start, unstuck_start + 1]].copy(),)
        self.coin_overrides = torch.as_tensor(
            self._prepare_coin_overrides(coin_overrides), device=gpu_device()
        )
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
                float(bool(filter_by_min_effective_cost)),
                0.0,  # Runtime factual-ring capacity; absent from compile identity.
                0.0,  # Internal factual evaluation control, independently of capture.
            ],
            dtype=torch.float32,
            device=gpu_device(),
        )
        self._buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self._recovery_buffers: dict[int, torch.Tensor] = {}
        self._volume_buffers: dict[int, torch.Tensor] = {}
        self._weighted_equity_buffers: dict[int, torch.Tensor] = {}
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
            ),
            dtype=np.float32,
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
        values = np.asarray(end_steps, dtype=np.int32)
        if values.shape != (batch_size,):
            raise ValueError(
                f"expected one multi-coin end step per candidate, got {values.shape}"
            )
        values = np.clip(values, 1, self.n - 1)
        return torch.as_tensor(
            np.ascontiguousarray(values), dtype=torch.int32, device=gpu_device()
        )

    def _dispatch(
        self,
        library,
        params_mps,
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
        volume_samples,
        weighted_equity_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.hour_log_ranges,
            self.coin_settings,
            self.coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
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
        if self.weighted_volume_enabled:
            kernel_args += (volume_samples,)
        if self.weighted_equity_cols:
            kernel_args += (weighted_equity_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        if self.unstuck_pnl_capacity:
            kernel_args += self._unstuck_history_buffers(batch_size)
        library.passivbot_ema_anchor_multicoin(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )

    def _library(self):
        loader, args = self._library_cache_call()
        return loader(*args)

    def _use_disabled_hsl_specialization(self, matrix):
        if getattr(self, "hsl_fact_capacity", 0):
            return False
        # Coin-mode forced delists still report per-coin panic segments when
        # HSL is disabled. The compact layout preserves only aggregate state.
        # Fused and TM kernels do not implement this one-side EMA layout.
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS
        return bool(
            self.coin_override_label == "EMA"
            and getattr(self, "hsl_disabled_specialization", True)
            and not self.coin_hsl_may_enable
            and np.all(matrix[:, keys.index("hsl_enabled")] <= 0.5)
            and np.isin(matrix[:, keys.index("hsl_signal_mode")], [0, 1]).all()
        )

    def _unstuck_ema_required(self, matrix):
        if not getattr(self, "unstuck_ema_specialization", True):
            return True
        keys = (TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
                if self.coin_override_label == "Trailing Martingale"
                else EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        return unstuck_ema_required(matrix, keys, self._unstuck_ema_overrides)

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
            self.unstuck_pnl_lookback_bars,
            self.unstuck_pnl_capacity,
            self.weighted_volume_enabled,
            self.raw_strategy_risk_enabled,
            self.raw_strategy_growth_enabled,
            self.weighted_raw_equity_enabled,
            self.weighted_account_equity_enabled,
            self.hsl_raw_tail_capacity,
            getattr(self, "dispatch_unstuck_ema_enabled", True),
            int(bool(getattr(self, "hsl_fact_capacity", 0))),
        )

    def _hsl_history_bytes_per_candidate(self):
        if not self.hsl_capacity:
            return 0
        _, nodes = _hsl_layout(self.hsl_capacity, getattr(self, "hsl_fact_capacity", 0))
        return self.hsl_scopes * (nodes * 32 + self.hsl_capacity * 8)

    def _history_bytes_per_candidate(self):
        return (
            self._hsl_history_bytes_per_candidate() + self.unstuck_pnl_capacity * 16
            + (4 * self.n_days if self.raw_strategy_risk_enabled else 0)
            + (8 * self.n_days if self.raw_strategy_growth_enabled else 0)
            + weighted_equity_history_bytes(
                self.n, self.n_days, self.weighted_equity_metrics
            )
            + (_recovery_history_bytes(self.n_recovery_samples)
               if self.recovery_distribution_enabled else 0)
            + (_volume_history_bytes(self.n)
               if getattr(self, "weighted_volume_enabled", False) else 0)
        )

    def _unstuck_history_buffers(self, batch_size):
        if (
            batch_size * self._history_bytes_per_candidate()
            > self.hsl_scratch_budget_bytes
        ):
            raise ValueError("GPU history batch exceeds its scratch budget")
        if batch_size not in self._unstuck_pnl_buffers:
            shape = (batch_size, self.unstuck_pnl_capacity, 2)
            self._unstuck_pnl_buffers = {
                batch_size: (
                    torch.empty(shape, dtype=torch.float32, device=gpu_device()),
                    torch.empty(shape, dtype=torch.int32, device=gpu_device()),
                )
            }
        return self._unstuck_pnl_buffers[batch_size]

    def _hsl_buffers(self, batch_size):
        _, nodes = _hsl_layout(self.hsl_capacity, getattr(self, "hsl_fact_capacity", 0))
        if (
            batch_size * self._history_bytes_per_candidate()
            > self.hsl_scratch_budget_bytes
        ):
            raise ValueError("GPU history batch exceeds its scratch budget")
        key = (batch_size, getattr(self, "hsl_fact_capacity", 0))
        if key not in self._hsl_scratch_buffers:
            self._hsl_scratch_buffers = {
                key: (
                    torch.empty(
                        (batch_size, self.hsl_scopes, nodes, 32),
                        dtype=torch.uint8,
                        device=gpu_device(),
                    ),
                    torch.empty(
                        (batch_size, self.hsl_scopes, self.hsl_capacity * 2),
                        dtype=torch.int32,
                        device=gpu_device(),
                    ),
                )
            }
        return self._hsl_scratch_buffers[key]

    def _run_history_batches(self, params, *, profile, end_steps):
        limit = self.hsl_scratch_budget_bytes // self._history_bytes_per_candidate()
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
        return _decode_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled)

    def _recovery_sample_buffer(self, batch_size: int):
        if batch_size * self._history_bytes_per_candidate() > self.hsl_scratch_budget_bytes:
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

    def _weighted_equity_sample_buffer(self, batch_size):
        if not self.weighted_equity_cols:
            return None
        if batch_size * self._history_bytes_per_candidate() > self.hsl_scratch_budget_bytes:
            raise ValueError("GPU weighted equity exceeds its scratch budget")
        if batch_size not in self._weighted_equity_buffers:
            self._weighted_equity_buffers = {
                batch_size: torch.empty(
                    (batch_size, self.n, self.weighted_equity_cols),
                    dtype=torch.float32, device=gpu_device(),
                )
            }
        samples = self._weighted_equity_buffers[batch_size]
        samples.fill_(float("nan"))
        return samples

    def _reduce_weighted_equity(self, samples, output):
        if not self.weighted_equity_cols:
            return {}
        interval = self.run_config.interval_ms
        first = output["first_eq_ts"].to(torch.float64)
        last = output["last_eq_ts"].to(torch.float64)
        observed = torch.isfinite(first)
        if bool((observed != torch.isfinite(last)).any()):
            raise RuntimeError("GPU weighted equity observation clocks disagree")
        # Relative f32 millisecond clocks can lose low bits over long histories.
        # Restore their integer bar grid before adding the absolute UTC origin.
        first_step = torch.round(torch.where(observed, first, 0) / interval)
        last_step = torch.round(torch.where(observed, last, 0) / interval)
        counts = torch.where(observed, last_step - first_step + 1, 0).to(torch.long)
        timestamps = first_step * interval + self.run_config.first_ts_ms
        kwargs = dict(
            first_timestamps_ms=timestamps, sample_counts=counts,
            interval_ms=interval, n_days=self.n_days,
        )
        metrics = {}
        if self.weighted_raw_equity_enabled:
            metrics.update(weighted_equity_from_samples(
                samples[:, :, 0], curve="raw_strategy",
                requested=self.weighted_equity_metrics & WEIGHTED_RAW_EQUITY_METRICS,
                **kwargs,
            ))
        if self.weighted_account_equity_enabled:
            metrics.update(weighted_equity_from_samples(
                samples[:, :, int(self.weighted_raw_equity_enabled)], curve="account",
                fill_counts=output["fill_count"],
                requested=self.weighted_equity_metrics & WEIGHTED_ACCOUNT_EQUITY_METRICS,
                **kwargs,
            ))
        return metrics

    def _volume_sample_buffer(self, batch_size: int):
        if not self.weighted_volume_enabled:
            return None
        if batch_size * self._history_bytes_per_candidate() > self.hsl_scratch_budget_bytes:
            raise ValueError("GPU volume history batch exceeds its scratch budget")
        if batch_size not in self._volume_buffers:
            self._volume_buffers = {
                batch_size: torch.zeros(
                    (batch_size, self.n, 2), dtype=torch.float32, device=gpu_device(),
                )
            }
        else:
            self._volume_buffers[batch_size].zero_()
        return self._volume_buffers[batch_size]

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
                    (batch_size, MPS_ENTRY_INTERVAL_COUNT_COLS),
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
        if not getattr(self, "native_factual_hsl", False):
            return
        keys = (TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
                if self.coin_override_label == "Trailing Martingale"
                else EMA_ANCHOR_MULTICOIN_PARAM_KEYS)
        matrix = np.asarray(params, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(keys) * self.hsl_replay_sides:
            raise ValueError("invalid native GPU candidate parameter shape")
        active = bool(np.any(self._native_hsl_effective_enablement(matrix, keys)))
        capacity = self.hsl_fact_capacity_learned if active else 0
        if capacity != self.hsl_fact_capacity:
            self._hsl_scratch_buffers.clear()
        self.hsl_fact_capacity = capacity
        self.hsl_factual_replay = active

    def run(self, params, *, profile=False, end_steps=None):
        """Grow bounded factual storage from rejected GPU work, without publishing it."""
        self._prepare_native_factual_hsl(params)
        count_before = getattr(self, "hsl_fact_retry_count_total", 0)
        seconds_before = getattr(self, "hsl_fact_retry_seconds_total", 0.0)
        while True:
            started = time.perf_counter()
            try:
                output = self._run_factual_attempt(params, profile=profile, end_steps=end_steps)
            except HslFactHistoryOverflow:
                capacity = getattr(self, "hsl_fact_capacity", 0)
                if capacity < 1:
                    raise
                self.interrupt_check()
                self.hsl_fact_capacity = capacity * 2
                if self._history_bytes_per_candidate() > self.hsl_scratch_budget_bytes:
                    self.hsl_fact_capacity = capacity
                    raise HslFactHistoryOverflow(
                        "GPU HSL factual history overflow exceeds single-candidate scratch budget"
                    ) from None
                # The rejected attempt completed before decoding. Its histories
                # have no authority; a fresh replay resets all device cursors.
                self._hsl_scratch_buffers.clear()
                if getattr(self, "native_factual_hsl", False):
                    self.hsl_fact_capacity_learned = self.hsl_fact_capacity
                self.hsl_fact_retry_count_total = getattr(self, "hsl_fact_retry_count_total", 0) + 1
                self.hsl_fact_retry_seconds_total = (getattr(self, "hsl_fact_retry_seconds_total", 0.0)
                                                     + time.perf_counter() - started)
                continue
            self.last_hsl_fact_retries = getattr(self, "hsl_fact_retry_count_total", 0) - count_before
            if profile and self.last_hsl_fact_retries:
                self.last_profile["hsl_fact_retry_count"] = self.last_hsl_fact_retries
                self.last_profile["hsl_fact_retry_seconds"] = (self.hsl_fact_retry_seconds_total
                                                                - seconds_before)
            return output

    def _run_factual_attempt(
        self,
        params: np.ndarray,
        *,
        profile: bool = False,
        end_steps: np.ndarray | None = None,
    ) -> dict:
        if self._history_bytes_per_candidate():
            split = self._run_history_batches(
                params, profile=profile, end_steps=end_steps
            )
            if split is not None:
                return split
        if self.hsl_capacity:
            keys = (
                TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
                if self.coin_override_label == "Trailing Martingale"
                else EMA_ANCHOR_MULTICOIN_PARAM_KEYS
            )
            policy_matrix = np.asarray(params, dtype=np.float32)
            effective = (self._native_hsl_effective_enablement(policy_matrix, keys)
                         if getattr(self, "native_factual_hsl", False) else None)
            if policy_matrix.ndim == 2 and policy_matrix.shape[1] == len(keys):
                policy_matrix = np.concatenate((policy_matrix, policy_matrix), axis=1)
                if effective is not None:
                    effective = np.repeat(effective, 2, axis=1)
            MpsEmaAnchorRunner._validate_hsl_params(
                self, policy_matrix, keys, effective_enabled=effective)
        started = time.perf_counter() if profile else 0.0
        self.settings[-2] = getattr(self, "hsl_fact_capacity", 0)
        factual_replay = bool(getattr(self, "hsl_factual_replay", False))
        if factual_replay and not getattr(self, "hsl_fact_capacity", 0):
            raise ValueError("GPU factual HSL evaluation requires retained fill capacity")
        self.settings[-1] = float(factual_replay)
        matrix = self._pack_params(params)
        self.dispatch_hsl_disabled = self._use_disabled_hsl_specialization(matrix)
        self.dispatch_unstuck_ema_enabled = self._unstuck_ema_required(matrix)
        packed = time.perf_counter() if profile else 0.0
        params_mps = torch.as_tensor(matrix, device=gpu_device())
        batch_size = int(matrix.shape[0])
        end_steps_mps = self._end_steps(end_steps, batch_size)
        daily, scalars, gaps, coin_fill_counts = self._output_buffers(batch_size)
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
        volume_samples = self._volume_sample_buffer(batch_size)
        weighted_equity_samples = self._weighted_equity_sample_buffer(batch_size)
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
        loader, library_args = self._library_cache_call()
        library, cold = _cached_library_with_miss(loader, *library_args)
        compiled = time.perf_counter() if profile else 0.0
        if profile:
            synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled
        self._dispatch(
            library,
            params_mps,
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
            volume_samples,
            weighted_equity_samples,
            batch_size=batch_size,
        )
        if profile:
            synchronize()
            finished = time.perf_counter()
            self.last_profile = {
                "cpu_pack_seconds": packed - started,
                "upload_and_zero_seconds": prepared - packed,
                "compile_seconds": compiled - prepared,
                "pre_dispatch_sync_seconds": dispatched - compiled,
                "kernel_seconds": finished - dispatched,
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
        if self.raw_strategy_risk_enabled:
            output["raw_strategy_day_max_dd"] = daily[:, :, -1]
        if self.raw_strategy_growth_enabled:
            offset = self.daily_cols - int(self.raw_strategy_risk_enabled) - 2
            output["raw_strategy_day_end_eq"] = daily[:, :, offset]
            output["raw_strategy_day_min_eq"] = daily[:, :, offset + 1]
        output.update(self._reduce_weighted_equity(weighted_equity_samples, output))
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        output.update(
            _decode_entry_interval_outputs(entry_interval_stats, entry_interval_counts)
        )
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        if self.weighted_volume_enabled:
            output["volume_pct_per_day_avg_w"] = weighted_volume_from_samples(
                volume_samples, output["first_eq_ts"], output["last_eq_ts"],
                start_minute_of_day=self.start_minute_of_day,
                interval_minutes=self.interval_minutes,
            )
        if self.collect_coin_fill_counts:
            output["coin_fill_counts"] = coin_fill_counts
        if profile:
            synchronize()
            self.last_profile["metric_decode_seconds"] = time.perf_counter() - finished
        return output


class MpsEmaAnchorMulticoinFusedRunner(MpsEmaAnchorMulticoinRunner):
    """Persistent dual-side shared-account EMA Anchor runner on Apple MPS."""

    scalar_cols = MPS_MULTICOIN_FUSED_SCALAR_COLS
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
        filter_by_min_effective_cost: bool = False,
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        weighted_volume_enabled: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        weighted_equity_metrics=(),
        hedge_mode: bool = True,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        unstuck_pnl_lookback_bars: int = 0,
        factual_hsl: bool = False,
        interrupt_check=None,
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
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market_long,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            hsl_raw_tail_enabled=hsl_raw_tail_enabled,
            recovery_distribution_enabled=recovery_distribution_enabled,
            weighted_volume_enabled=weighted_volume_enabled,
            raw_strategy_risk_enabled=raw_strategy_risk_enabled,
            raw_strategy_growth_enabled=raw_strategy_growth_enabled,
            weighted_equity_metrics=weighted_equity_metrics,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
            pnl_lookback_bars=pnl_lookback_bars,
            unstuck_pnl_lookback_bars=unstuck_pnl_lookback_bars,
            factual_hsl=factual_hsl,
            interrupt_check=interrupt_check,
        )
        if short_coin_overrides is None:
            short_coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols),
                np.nan,
                dtype=np.float32,
            )
        short_coin_overrides = np.asarray(short_coin_overrides, dtype=np.float32)
        expected_shape = (self.n_coins, self.coin_override_cols)
        if short_coin_overrides.shape != expected_shape:
            raise ValueError(
                "expected fused multicoin EMA short override matrix shaped "
                f"{expected_shape}, got {short_coin_overrides.shape}"
            )
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
                float(bool(filter_by_min_effective_cost)),
                0.0,  # Runtime factual-ring capacity; absent from compile identity.
                0.0,  # Internal factual evaluation control, independently of capture.
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
            ),
            dtype=np.float32,
        )

    def _dispatch(
        self,
        library,
        params_mps,
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
        volume_samples,
        weighted_equity_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.hour_log_ranges,
            self.coin_settings,
            self.coin_overrides,
            self.short_coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
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
        if self.weighted_volume_enabled:
            kernel_args += (volume_samples,)
        if self.weighted_equity_cols:
            kernel_args += (weighted_equity_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        if self.unstuck_pnl_capacity:
            kernel_args += self._unstuck_history_buffers(batch_size)
        library.passivbot_ema_anchor_multicoin_fused(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled)


class MpsEmaAnchorMulticoinLongRunner(MpsEmaAnchorMulticoinRunner):
    """Compatibility wrapper for the original long-only multicoin runner."""

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
    ):
        super().__init__(
            run,
            data,
            side="long",
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
        )


class MpsEmaAnchorMulticoinShortRunner(MpsEmaAnchorMulticoinRunner):
    """Short-only multicoin EMA Anchor screening runner."""

    def __init__(
        self,
        run: ProxyRun,
        data: dict,
        *,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
    ):
        super().__init__(
            run,
            data,
            side="short",
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
        )


class MpsTrailingMartingaleMulticoinRunner(MpsEmaAnchorMulticoinRunner):
    """Persistent single-side multi-coin Trailing Martingale proxy on MPS."""

    replay_kernel_name = "passivbot_trailing_martingale_multicoin"
    replay_state_size_kernel_name = "passivbot_tm_multicoin_replay_state_bytes"
    replay_sides = 1

    coin_override_cols = TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    coin_override_label = "Trailing Martingale"

    def _prepare_coin_overrides(self, coin_overrides: np.ndarray) -> np.ndarray:
        return _scale_tm_multicoin_coin_overrides(coin_overrides, self.interval_minutes)

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
        filter_by_min_effective_cost: bool = False,
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        weighted_volume_enabled: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        weighted_equity_metrics=(),
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        unstuck_pnl_lookback_bars: int = 0,
        max_dispatch_candidate_bars: int | None = None,
        factual_hsl: bool = False,
        interrupt_check=None,
    ):
        if max_dispatch_candidate_bars is not None and max_dispatch_candidate_bars <= 0:
            raise ValueError("max_dispatch_candidate_bars must be positive")
        self.max_dispatch_candidate_bars = max_dispatch_candidate_bars
        self._replay_state_bytes = None
        self._replay_state_sizes = {}
        self._replay_states = {}
        self._last_temporal_dispatch = None
        self.loss_gate_enabled = _encode_max_realized_loss_pct(max_realized_loss_pct) < 1.0
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
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            hsl_raw_tail_enabled=hsl_raw_tail_enabled,
            recovery_distribution_enabled=recovery_distribution_enabled,
            weighted_volume_enabled=weighted_volume_enabled,
            raw_strategy_risk_enabled=raw_strategy_risk_enabled,
            raw_strategy_growth_enabled=raw_strategy_growth_enabled,
            weighted_equity_metrics=weighted_equity_metrics,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
            pnl_lookback_bars=pnl_lookback_bars,
            unstuck_pnl_lookback_bars=unstuck_pnl_lookback_bars,
            factual_hsl=factual_hsl,
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
        )
        return _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, sides=1
        )

    def _library(self):
        loader, args = self._library_cache_call()
        return loader(*args)

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
            self.unstuck_pnl_lookback_bars,
            self.unstuck_pnl_capacity,
            self.loss_gate_specialization and not self.loss_gate_enabled,
            self.weighted_volume_enabled,
            self.raw_strategy_risk_enabled,
            self.raw_strategy_growth_enabled,
            self.weighted_raw_equity_enabled,
            self.weighted_account_equity_enabled,
            self.hsl_raw_tail_capacity,
            getattr(self, "dispatch_unstuck_ema_enabled", True),
            int(bool(getattr(self, "hsl_fact_capacity", 0))),
        )
        return _trailing_martingale_multicoin_shader_library, args

    def _dispatch(
        self,
        library,
        params_mps,
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
        volume_samples,
        weighted_equity_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.touch_nearest_ticks,
            self.touch_min_qty_bits,
            self.touch_min_qty_relation,
            self.hour_log_ranges,
            self.coin_settings,
            self.coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
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
        if self.weighted_volume_enabled:
            kernel_args += (volume_samples,)
        if self.weighted_equity_cols:
            kernel_args += (weighted_equity_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        if self.unstuck_pnl_capacity:
            kernel_args += self._unstuck_history_buffers(batch_size)
        self._dispatch_replay(library, kernel_args, end_steps, batch_size=batch_size)

    def _dispatch_replay(self, library, kernel_args, end_steps, *, batch_size):
        kernel = getattr(library, self.replay_kernel_name)
        if self.max_dispatch_candidate_bars is None:
            dispatch_options = {"threads": (batch_size, 1, 1)}
            if gpu_device(torch) == "cuda" and self.replay_sides == 1:
                # Spread independent, state-heavy candidates across more SMs.
                dispatch_options["group_size"] = (32, 1, 1)
            kernel(
                *kernel_args, **dispatch_options
            )
            return
        chunk_bars = min(
            MPS_TM_MULTICOIN_CHUNK_BARS,
            MPS_TM_MULTICOIN_CHUNK_CANDIDATE_STEPS // batch_size,
            self.max_dispatch_candidate_bars // (batch_size * self.n_coins * self.replay_sides),
        )
        if chunk_bars < 1:
            raise ValueError("MPS replay batch exceeds the per-dispatch work envelope")
        _, library_args = self._library_cache_call()
        if library_args not in self._replay_state_sizes:
            size = torch.empty(1, dtype=torch.int32, device=gpu_device())
            getattr(library, self.replay_state_size_kernel_name)(size, threads=1)
            self._replay_state_sizes[library_args] = int(size.item())
        self._replay_state_bytes = self._replay_state_sizes[library_args]
        state_key = (batch_size, self._replay_state_bytes)
        if state_key not in self._replay_states:
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
        replay_progress = TemporalReplayProgress(batch_size, stop_k - 1, history_chunk_bars=chunk_bars)
        replay_started = time.perf_counter()
        next_progress = replay_started + 30.0
        # One SIMD-width group distributes independent, state-heavy replays
        # across GPU cores instead of packing the batch into a large group.
        threads_per_threadgroup = min(batch_size, 32)
        for begin_k in range(1, max(2, stop_k), chunk_bars):
            self.interrupt_check()
            replay_range = torch.tensor(
                [begin_k, min(begin_k + chunk_bars, stop_k)],
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
            processed_bars = min(begin_k + chunk_bars, stop_k) - begin_k
            record_replay_chunk(
                batch_size, processed_bars, stop_k - 1, dispatch_seconds,
                eligible=begin_k > 1 and processed_bars == chunk_bars,
            )
            now = time.perf_counter()
            completed_k = min(begin_k + chunk_bars, stop_k)
            if now >= next_progress and completed_k < stop_k:
                replay_progress.log("progress", completed_k - 1, now - replay_started, kernel_dispatches=dispatch_count)
                next_progress = now + 30.0
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

    def run(self, params, *, profile=False, end_steps=None):
        self._last_temporal_dispatch = None
        output = super().run(params, profile=profile, end_steps=end_steps)
        if (
            profile
            and self._last_temporal_dispatch is not None
            and "candidate_batch_count" not in self.last_profile
        ):
            self.last_profile.update(self._last_temporal_dispatch)
        return output

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled)


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
        filter_by_min_effective_cost: bool = False,
        market_order_slippage_pct: float = 0.0,
        market_orders_allowed: bool = False,
        market_order_near_touch_threshold: float = 0.001,
        hsl_panic_market_long: bool = False,
        hsl_panic_market_short: bool = False,
        hsl_ema_tail_enabled: bool = False,
        hsl_raw_drawdown_enabled: bool = False,
        hsl_raw_tail_enabled: bool = False,
        recovery_distribution_enabled: bool = False,
        weighted_volume_enabled: bool = False,
        raw_strategy_risk_enabled: bool = False,
        raw_strategy_growth_enabled: bool = False,
        weighted_equity_metrics=(),
        hedge_mode: bool = True,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
        pnl_lookback_bars: int = 0,
        unstuck_pnl_lookback_bars: int = 0,
        max_dispatch_candidate_bars: int | None = None,
        factual_hsl: bool = False,
        interrupt_check=None,
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
            market_order_slippage_pct=market_order_slippage_pct,
            market_orders_allowed=market_orders_allowed,
            market_order_near_touch_threshold=market_order_near_touch_threshold,
            hsl_panic_market=hsl_panic_market_long,
            hsl_ema_tail_enabled=hsl_ema_tail_enabled,
            hsl_raw_drawdown_enabled=hsl_raw_drawdown_enabled,
            hsl_raw_tail_enabled=hsl_raw_tail_enabled,
            recovery_distribution_enabled=recovery_distribution_enabled,
            weighted_volume_enabled=weighted_volume_enabled,
            raw_strategy_risk_enabled=raw_strategy_risk_enabled,
            raw_strategy_growth_enabled=raw_strategy_growth_enabled,
            weighted_equity_metrics=weighted_equity_metrics,
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
            pnl_lookback_bars=pnl_lookback_bars,
            unstuck_pnl_lookback_bars=unstuck_pnl_lookback_bars,
            max_dispatch_candidate_bars=max_dispatch_candidate_bars,
            factual_hsl=factual_hsl,
            interrupt_check=interrupt_check,
        )
        if short_coin_overrides is None:
            short_coin_overrides = np.full(
                (self.n_coins, self.coin_override_cols),
                np.nan,
                dtype=np.float32,
            )
        short_coin_overrides = np.asarray(short_coin_overrides, dtype=np.float32)
        expected_shape = (self.n_coins, self.coin_override_cols)
        if short_coin_overrides.shape != expected_shape:
            raise ValueError(
                "expected fused multicoin Trailing Martingale short override "
                f"matrix shaped {expected_shape}, got {short_coin_overrides.shape}"
            )
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
        self.short_coin_overrides = torch.as_tensor(
            self._prepare_coin_overrides(short_coin_overrides), device=gpu_device()
        )
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            float(max_realized_loss_pct)
        )
        liq_floor = max(0.0, run.starting_balance) * max(0.0, run.liquidation_threshold)
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
                float(bool(filter_by_min_effective_cost)),
                0.0,  # Runtime factual-ring capacity; absent from compile identity.
                0.0,  # Internal factual evaluation control, independently of capture.
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
        )
        return _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, sides=2
        )

    def _dispatch(
        self,
        library,
        params_mps,
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
        volume_samples,
        weighted_equity_samples,
        *,
        batch_size: int,
    ) -> None:
        kernel_args = (
            self.bars,
            self.fill_ticks,
            self.touch_ticks,
            self.touch_nearest_ticks,
            self.touch_min_qty_bits,
            self.touch_min_qty_relation,
            self.hour_log_ranges,
            self.coin_settings,
            self.coin_overrides,
            self.short_coin_overrides,
            params_mps,
            self.settings,
            sizes,
            end_steps,
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
        if self.weighted_volume_enabled:
            kernel_args += (volume_samples,)
        if self.weighted_equity_cols:
            kernel_args += (weighted_equity_samples,)
        if self.hsl_capacity:
            kernel_args += self._hsl_buffers(batch_size)
        if self.unstuck_pnl_capacity:
            kernel_args += self._unstuck_history_buffers(batch_size)
        self._dispatch_replay(library, kernel_args, end_steps, batch_size=batch_size)

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps, btc_risk_enabled=self.btc_risk_enabled)


class MpsTrailingMartingaleRunner(MpsEmaAnchorRunner):
    """Persistent single-coin trailing-martingale runner on Apple MPS."""

    def __init__(
        self,
        *args,
        hsl_enabled: bool = True,
        hsl_diagnostics_enabled: bool = True,
        entry_interval_enabled: bool = False,
        max_dispatch_candidate_bars: int | None = None,
        interrupt_check=None,
        **kwargs,
    ):
        super().__init__(*args, hsl_enabled=hsl_enabled, **kwargs)
        if max_dispatch_candidate_bars is not None:
            if max_dispatch_candidate_bars <= 0:
                raise ValueError("max_dispatch_candidate_bars must be positive")
        self.max_dispatch_candidate_bars = max_dispatch_candidate_bars
        self.tuning_chunk_bars = None
        self.interrupt_check = interrupt_check or (lambda: None)
        self._replay_state_sizes = {}
        self._replay_states = {}
        self._encode_hour_boundary_flags()
        self.hsl_diagnostics_enabled = bool(hsl_diagnostics_enabled)
        if not self.hsl_diagnostics_enabled and (
            self.hsl_ema_tail_enabled
            or self.hsl_raw_drawdown_enabled
            or self.hsl_raw_tail_enabled
        ):
            raise ValueError("HSL diagnostic feature outputs require diagnostics")
        self.entry_interval_enabled = bool(entry_interval_enabled)
        self._entry_interval_stat_buffers: dict[int, torch.Tensor] = {}
        self._entry_interval_count_buffers: dict[int, torch.Tensor] = {}
        self.shader_topology = single_coin_shader_topology(
            long_enabled=self.long_enabled,
            short_enabled=self.short_enabled,
            hsl_enabled=bool(hsl_enabled),
            hsl_one_side_enabled=True,
        )
        self.shader_topology = "generic"
        # The specialized no-HSL kernels retain the original 66-column ABI.
        # Every EMA-tail metric is identically zero when HSL is disabled, so
        # keep that faster topology and let the decoder synthesize zeroes.
        if self.shader_topology.endswith("_no_hsl"):
            self.hsl_ema_tail_enabled = False
            self.hsl_raw_drawdown_enabled = False
            self.hsl_raw_tail_enabled = False

    def _encode_hour_boundary_flags(self) -> None:
        first_ts_ms = int(self.run_config.first_ts_ms)
        interval_ms = int(self.run_config.interval_ms)
        derived_timestamps = (
            first_ts_ms + np.arange(self.n, dtype=np.int64) * interval_ms
        )
        hour_indices = derived_timestamps // 3_600_000
        boundary_indices = np.flatnonzero(
            np.r_[False, hour_indices[1:] > hour_indices[:-1]]
        )
        hour_boundary_bits = np.zeros(self.n, dtype=np.int32)
        last_hour_boundary_ms = (first_ts_ms // 3_600_000) * 3_600_000
        for step in boundary_indices:
            current_ts_ms = int(derived_timestamps[step])
            window_start_ms = max(first_ts_ms, last_hour_boundary_ms)
            window_ready = current_ts_ms > window_start_ms + interval_ms
            current_hour_boundary_ms = (current_ts_ms // 3_600_000) * 3_600_000
            next_window_start = max(
                0,
                (current_hour_boundary_ms - first_ts_ms) // interval_ms,
            )
            hour_boundary_bits[step] = (
                2 | (4 if window_ready else 0) | (8 if next_window_start < step else 0)
            )
            last_hour_boundary_ms = current_hour_boundary_ms
        boundary_bits = torch.as_tensor(
            hour_boundary_bits, dtype=torch.int32, device=gpu_device()
        )
        self.flags[:, 3].bitwise_or_(boundary_bits)

    def _shader_library(self):
        loader, args = self._shader_library_cache_call()
        return loader(*args)

    def _shader_library_cache_call(
        self,
        dispatch_features: (
            tuple[bool, bool, bool, bool, bool, bool, bool] | None
        ) = None,
        *,
        temporal_chunking: bool | None = None,
    ):
        if dispatch_features is None:
            dispatch_features = (
                False,
                False,
                False,
                False,
                not getattr(self, "market_orders_allowed", False),
                not getattr(self, "loss_gate_enabled", False),
                False,
            )
        if self.shader_topology == "long_hsl":
            return _trailing_martingale_long_hsl_shader_library, (
                *dispatch_features,
                self.hsl_ema_tail_enabled,
                self.hsl_raw_drawdown_enabled,
                self.hsl_raw_tail_enabled,
                self.recovery_distribution_enabled,
                self.btc_risk_enabled,
                self.equity_balance_diff_enabled,
                self.entry_interval_enabled,
                self.hsl_diagnostics_enabled,
                self.hsl_raw_tail_capacity,
            )
        if self.shader_topology == "short_hsl":
            return _trailing_martingale_short_hsl_shader_library, (
                *dispatch_features,
                self.hsl_ema_tail_enabled,
                self.hsl_raw_drawdown_enabled,
                self.hsl_raw_tail_enabled,
                self.recovery_distribution_enabled,
                self.btc_risk_enabled,
                self.equity_balance_diff_enabled,
                self.entry_interval_enabled,
                self.hsl_diagnostics_enabled,
                self.hsl_raw_tail_capacity,
            )
        if self.shader_topology == "long_no_hsl":
            return _trailing_martingale_long_no_hsl_shader_library, (
                *dispatch_features,
                self.recovery_distribution_enabled,
                self.btc_risk_enabled,
                self.equity_balance_diff_enabled,
                self.entry_interval_enabled,
            )
        if self.shader_topology == "short_no_hsl":
            return _trailing_martingale_short_no_hsl_shader_library, (
                *dispatch_features,
                self.recovery_distribution_enabled,
                self.btc_risk_enabled,
                self.equity_balance_diff_enabled,
                self.entry_interval_enabled,
            )
        return _trailing_martingale_shader_library, (
            *dispatch_features,
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.hsl_raw_tail_enabled,
            self.recovery_distribution_enabled,
            self.btc_risk_enabled,
            self.equity_balance_diff_enabled,
            self.entry_interval_enabled,
            self.hsl_diagnostics_enabled,
            (
                (self.max_dispatch_candidate_bars is not None)
                if temporal_chunking is None
                else temporal_chunking
            ),
            self.hsl_capacity,
            self.hsl_raw_tail_capacity,
        )

    def _entry_interval_buffers(self, batch_size: int):
        if not self.entry_interval_enabled:
            return None, None
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
                    (batch_size, MPS_ENTRY_INTERVAL_COUNT_COLS),
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

    def _pack_params(self, params: np.ndarray) -> np.ndarray:
        params = _upgrade_legacy_single_coin_wel_params(
            params,
            side_width=len(TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS),
        )
        expected = len(TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS) * 2
        if params.ndim != 2 or params.shape[1] != expected:
            got = params.shape[1] if params.ndim == 2 else params.shape
            raise ValueError(
                "expected directional trailing-martingale parameter matrix with "
                f"{expected} columns, got {got}"
            )
        scaled = _scale_single_coin_minute_parameters(
            params,
            TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS,
            sides=2,
            interval_minutes=self.interval_minutes,
        )
        packed = _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS, sides=2
        )
        self._validate_hsl_params(packed, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS)
        return packed

    def _trailing_single_coin_size_values(
        self,
        batch_size: int,
        parameter_count: int,
        *,
        end_step: int | None = None,
        history_start_step: int | None = None,
        trade_start_step: int | None = None,
    ) -> list[int]:
        values = self._single_coin_size_values(
            batch_size, parameter_count, end_step=end_step
        )
        effective_end_step = self.n if end_step is None else int(end_step)
        bounded = history_start_step is not None or trade_start_step is not None
        if bounded and (history_start_step is None or trade_start_step is None):
            raise ValueError(
                "recent-history MPS dispatch requires both history_start_step "
                "and trade_start_step"
            )
        if bounded:
            history_start_step = int(history_start_step)
            trade_start_step = int(trade_start_step)
            if not 0 <= history_start_step < trade_start_step < effective_end_step - 1:
                raise ValueError(
                    "recent-history MPS steps must satisfy 0 <= history start < "
                    "trade start < end_step - 1"
                )
            interval_ms = int(self.run_config.interval_ms)
            first_ts_ms = int(self.run_config.first_ts_ms)
            seed_ts_ms = first_ts_ms + history_start_step * interval_ms
            next_hour_ms = (seed_ts_ms // 3_600_000 + 1) * 3_600_000
            first_hour_step = min(
                effective_end_step,
                int(np.ceil((next_hour_ms - first_ts_ms) / interval_ms)),
            )
            first_hour_ts_ms = first_ts_ms + first_hour_step * interval_ms
            first_hour_ready = int(
                first_hour_step < effective_end_step
                and first_hour_ts_ms > seed_ts_ms + interval_ms
            )
            first_hour_boundary_ms = (first_hour_ts_ms // 3_600_000) * 3_600_000
            first_next_window_start = max(
                history_start_step,
                history_start_step
                + (first_hour_boundary_ms - seed_ts_ms) // interval_ms,
            )
            recovery_sample_count = (
                min(
                    self.n_recovery_samples,
                    max(
                        1,
                        int(
                            np.ceil(
                                (effective_end_step - trade_start_step)
                                / self.recovery_stride
                            )
                        )
                        + 1,
                    ),
                )
                if self.recovery_distribution_enabled
                else 0
            )
        else:
            history_start_step = -1
            trade_start_step = -1
            first_hour_step = -1
            first_hour_ready = 0
            first_next_window_start = -1
            recovery_sample_count = (
                self.n_recovery_samples if self.recovery_distribution_enabled else 0
            )
        # Reserve the existing recovery ABI slots even when that feature is
        # compiled out, then append the recent-window fields at fixed indices.
        if not self.recovery_distribution_enabled:
            values.extend([0, 0])
        values.extend(
            [
                history_start_step,
                trade_start_step,
                recovery_sample_count,
                first_hour_step,
                first_hour_ready,
                first_next_window_start,
            ]
        )
        return values

    def run(
        self,
        params: np.ndarray,
        *,
        profile: bool = False,
        end_step: int | None = None,
        history_start_step: int | None = None,
        trade_start_step: int | None = None,
    ) -> dict:
        batched = self._run_hsl_batches(
            params,
            profile=profile,
            end_step=end_step,
            history_start_step=history_start_step,
            trade_start_step=trade_start_step,
        )
        if batched is not None:
            return batched
        started = time.perf_counter() if profile else 0.0
        matrix = self._pack_params(params)
        dispatch_features = _tm_dispatch_specialization(
            matrix,
            long_enabled=self.long_enabled,
            short_enabled=self.short_enabled,
            market_orders_allowed=self.market_orders_allowed,
            loss_gate_enabled=self.loss_gate_enabled,
        )
        packed = time.perf_counter() if profile else 0.0
        params_mps = torch.as_tensor(matrix, device=gpu_device())
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
        equity_balance_diff = self._equity_balance_diff_buffer(batch_size)
        entry_interval_stats, entry_interval_counts = self._entry_interval_buffers(
            batch_size
        )
        effective_end_step = self.n if end_step is None else int(end_step)
        effective_history_start = (
            -1 if history_start_step is None else int(history_start_step)
        )
        effective_trade_start = (
            -1 if trade_start_step is None else int(trade_start_step)
        )
        effective_recovery_sample_count = self.n_recovery_samples
        if self.recovery_distribution_enabled and effective_trade_start >= 0:
            effective_recovery_sample_count = min(
                self.n_recovery_samples,
                max(
                    1,
                    int(
                        np.ceil(
                            (effective_end_step - effective_trade_start)
                            / self.recovery_stride
                        )
                    )
                    + 1,
                ),
            )
        sizes_key = (
            batch_size,
            int(matrix.shape[1]),
            effective_end_step,
            effective_history_start,
            effective_trade_start,
        )
        if sizes_key not in self._sizes:
            size_values = self._trailing_single_coin_size_values(
                batch_size,
                int(matrix.shape[1]),
                end_step=effective_end_step,
                history_start_step=history_start_step,
                trade_start_step=trade_start_step,
            )
            self._sizes[sizes_key] = torch.tensor(
                size_values,
                dtype=torch.int32,
                device=gpu_device(),
            )
        prepared = time.perf_counter() if profile else 0.0
        # A small seed pool, cache miss, or final batch may fit the full-history
        # work envelope even when the configured batch ceiling needs chunking.
        # Preserve the cheaper unchunked shader for those actual dispatches.
        temporal_chunking = (
            self.max_dispatch_candidate_bars is not None
            and batch_size * 2 * (effective_end_step - max(0, effective_history_start))
            > self.max_dispatch_candidate_bars
        ) or (
            self.tuning_chunk_bars is not None
            and effective_end_step - max(0, effective_history_start)
            > self.tuning_chunk_bars
        )
        loader, library_args = self._shader_library_cache_call(
            dispatch_features, temporal_chunking=temporal_chunking
        )
        library, cold = _cached_library_with_miss(loader, *library_args)
        compiled = time.perf_counter() if profile else 0.0
        if profile:
            synchronize()
            dispatched = time.perf_counter()
        else:
            dispatched = compiled

        def dispatch_once():
            kernel_args = (
                self.bars,
                self.flags,
                params_mps,
                self.settings,
                self._sizes[sizes_key],
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
            )
            kernel_args += self._hsl_buffers(batch_size)
            if self.recovery_distribution_enabled:
                kernel_args += (recovery_samples,)
            if not temporal_chunking:
                library.passivbot_trailing_martingale(
                    *kernel_args,
                    threads=(batch_size, 1, 1),
                    **(
                        {"group_size": (min(batch_size, 64), 1, 1)}
                        if self.max_dispatch_candidate_bars is not None
                        else {}
                    ),
                )
                return {"dispatch_count": 1}
            chunk_bars = min(
                MPS_TM_SINGLE_COIN_CHUNK_BARS,
                (
                    self.max_dispatch_candidate_bars // (batch_size * 2)
                    if self.max_dispatch_candidate_bars is not None
                    else MPS_TM_SINGLE_COIN_CHUNK_BARS
                ),
                self.tuning_chunk_bars or MPS_TM_SINGLE_COIN_CHUNK_BARS,
            )
            if chunk_bars < 1:
                raise ValueError(
                    "MPS replay batch exceeds the per-dispatch work envelope"
                )
            if library_args not in self._replay_state_sizes:
                size = torch.empty(1, dtype=torch.int32, device=gpu_device())
                library.passivbot_tm_single_coin_replay_state_bytes(size, threads=1)
                self._replay_state_sizes[library_args] = int(size.item())
            # Feature specialization changes the state ABI. Only the current
            # candidate batch allocation is retained across evaluations.
            state_bytes = self._replay_state_sizes[library_args]
            state_key = (batch_size, state_bytes)
            if state_key not in self._replay_states:
                self._replay_states = {
                    state_key: torch.empty(
                        (batch_size, state_bytes),
                        dtype=torch.uint8,
                        device=gpu_device(),
                    )
                }
            replay_states = self._replay_states[state_key]
            begin = max(1, effective_history_start + 1)
            stop = effective_end_step - 1
            count = 0
            longest = 0.0
            replay_progress = TemporalReplayProgress(batch_size, max(0, stop - begin), history_chunk_bars=chunk_bars)
            replay_started = time.perf_counter()
            next_progress = replay_started + 30.0
            for first in range(begin, stop, chunk_bars):
                self.interrupt_check()
                replay_range = torch.tensor(
                    [first, min(first + chunk_bars, stop)],
                    dtype=torch.int32,
                    device=gpu_device(),
                )
                started = time.perf_counter()
                library.passivbot_trailing_martingale(
                    *kernel_args,
                    replay_states,
                    replay_range,
                    threads=(batch_size, 1, 1),
                    group_size=(min(batch_size, 64), 1, 1),
                )
                synchronize()
                dispatch_seconds = time.perf_counter() - started
                longest = max(longest, dispatch_seconds)
                processed_bars = min(first + chunk_bars, stop) - first
                record_replay_chunk(
                    batch_size, processed_bars, stop - begin, dispatch_seconds,
                    eligible=first > begin and processed_bars == chunk_bars,
                )
                count += 1
                now = time.perf_counter()
                if now >= next_progress and first + chunk_bars < stop:
                    replay_progress.log(
                        "progress", first + chunk_bars - begin, now - replay_started,
                        kernel_dispatches=count,
                    )
                    next_progress = now + 30.0
            self.interrupt_check()
            replay_progress.log("complete", max(0, stop - begin), time.perf_counter() - replay_started, kernel_dispatches=count)
            return {
                "dispatch_count": count,
                "temporal_chunk_bars": chunk_bars,
                "kernel_candidate_steps": batch_size * (stop - begin),
                "replay_state_bytes_per_candidate": state_bytes,
                "threads_per_threadgroup": min(batch_size, 64),
                "max_dispatch_seconds": longest,
            }

        dispatch_profile = dispatch_once()
        if profile:
            synchronize()
            finished = time.perf_counter()
            self.last_profile = {
                "cpu_pack_seconds": packed - started,
                "upload_and_zero_seconds": prepared - packed,
                "compile_seconds": compiled - prepared,
                "pre_dispatch_sync_seconds": dispatched - compiled,
                "kernel_seconds": finished - dispatched,
                "batch_size": batch_size,
                **dispatch_profile,
                "cold": cold,
                "effective_candle_count": effective_end_step
                - max(0, effective_history_start),
                "dispatch_specialization": {
                    "trailing_entry_only": dispatch_features[0],
                    "recursive_entry_only": dispatch_features[1],
                    "trailing_close_only": dispatch_features[2],
                    "reducers_disabled": dispatch_features[3],
                    "market_orders_disabled": dispatch_features[4],
                    "loss_gate_disabled": dispatch_features[5],
                    "volatility_disabled": dispatch_features[6],
                },
            }
        else:
            self.last_profile = {}
            wait_for_cuda_stream()
        output = _decode_directional_outputs(daily, scalars, gaps)
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        output.update(
            _decode_entry_interval_outputs(entry_interval_stats, entry_interval_counts)
        )
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples[
                :, :effective_recovery_sample_count
            ]
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        if profile:
            synchronize()
            self.last_profile["metric_decode_seconds"] = time.perf_counter() - finished
        return output
