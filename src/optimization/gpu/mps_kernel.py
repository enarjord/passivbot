from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
import torch

from optimization.gpu.model import (
    EMA_ANCHOR_COIN_OVERRIDE_COLS,
    EMA_ANCHOR_COIN_OVERRIDE_COOLDOWN_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_HSL_START_COLUMN,
    EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS,
    EMA_ANCHOR_MULTICOIN_PARAM_KEYS,
    EMA_ANCHOR_SINGLE_COIN_PARAM_KEYS,
    GAP_BINS,
    ProxyMarket,
    ProxyRun,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
    TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
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
MPS_MULTICOIN_BASE_SCALAR_COLS = 61
MPS_MULTICOIN_EMA_TAIL_SCALAR_COLS = 63
MPS_MULTICOIN_RAW_DRAWDOWN_SCALAR_COLS = 65
MPS_MULTICOIN_SCALAR_COLS = 67
MPS_DIRECTIONAL_BASE_SCALAR_COLS = 66
MPS_DIRECTIONAL_EMA_TAIL_SCALAR_COLS = 68
MPS_DIRECTIONAL_RAW_DRAWDOWN_SCALAR_COLS = 70
MPS_DIRECTIONAL_SCALAR_COLS = 72
MPS_MULTICOIN_FUSED_BASE_SCALAR_COLS = 66
MPS_MULTICOIN_FUSED_EMA_TAIL_SCALAR_COLS = 68
MPS_MULTICOIN_FUSED_RAW_DRAWDOWN_SCALAR_COLS = 70
MPS_MULTICOIN_FUSED_SCALAR_COLS = 72
# A 30-day coin-HSL lookback can legitimately contain slightly more than
# 2,048 completed round trips for high-cadence single-coin candidates. Metal
# coalesces every realized-PnL component from one candle into one ring event,
# so ladder fill multiplicity does not consume extra slots. Keep this bounded,
# but leave enough headroom for dense valid event-candle windows.
MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY = 8192
MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS = 7
MPS_EQUITY_BALANCE_DIFF_COLS = 12
MPS_ENTRY_INTERVAL_STAT_COLS = 2
MPS_ENTRY_INTERVAL_COUNT_COLS = 129

_HSL_EMA_TAIL_DEFINE = "#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 1\n"
_HSL_RAW_DRAWDOWN_DEFINE = "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n"
_HSL_RAW_TAIL_DEFINE = "#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 1\n"
_HSL_DIAGNOSTICS_DISABLE_DEFINE = (
    "#define PASSIVBOT_HSL_DIAGNOSTICS_ENABLED 0\n"
)
_RECOVERY_DISTRIBUTION_DEFINE = (
    "#define PASSIVBOT_STRATEGY_EQ_RECOVERY_DISTRIBUTION_ENABLED 1\n"
)
_FIXED_WEL_DENOMINATOR_DEFINE = (
    "#define PASSIVBOT_DYNAMIC_WEL_BY_TRADABILITY 0\n"
)
_BTC_RISK_DEFINE = "#define PASSIVBOT_BTC_RISK_ENABLED 1\n"
_EQUITY_BALANCE_DIFF_DEFINE = (
    "#define PASSIVBOT_EQUITY_BALANCE_DIFF_ENABLED 1\n"
)
_ENTRY_INTERVAL_DEFINE = "#define PASSIVBOT_ENTRY_INTERVAL_ENABLED 1\n"
_TM_TRAILING_ENTRY_ONLY_DEFINE = (
    "#define PASSIVBOT_TM_TRAILING_ENTRY_ONLY 1\n"
)
_TM_RECURSIVE_ENTRY_ONLY_DEFINE = (
    "#define PASSIVBOT_TM_RECURSIVE_ENTRY_ONLY 1\n"
)
_TM_TRAILING_CLOSE_ONLY_DEFINE = (
    "#define PASSIVBOT_TM_TRAILING_CLOSE_ONLY 1\n"
)
_TM_REDUCERS_DISABLED_DEFINE = "#define PASSIVBOT_TM_REDUCERS_DISABLED 1\n"
_TM_MARKET_ORDERS_DISABLED_DEFINE = (
    "#define PASSIVBOT_TM_MARKET_ORDERS_DISABLED 1\n"
)
_TM_LOSS_GATE_DISABLED_DEFINE = "#define PASSIVBOT_TM_LOSS_GATE_DISABLED 1\n"
_TM_VOLATILITY_DISABLED_DEFINE = (
    "#define PASSIVBOT_TM_VOLATILITY_DISABLED 1\n"
)


def _with_hsl_ema_tail(source: str, enabled: bool) -> str:
    if not enabled:
        return source
    if "#ifndef PASSIVBOT_HSL_EMA_TAIL_ENABLED" not in source:
        raise RuntimeError("MPS source is missing the HSL EMA-tail feature guard")
    return _HSL_EMA_TAIL_DEFINE + source


def _with_hsl_features(
    source: str,
    *,
    ema_tail_enabled: bool,
    raw_drawdown_enabled: bool,
    raw_tail_enabled: bool,
    diagnostics_enabled: bool = True,
) -> str:
    if raw_tail_enabled and not raw_drawdown_enabled:
        raise ValueError("HSL raw-tail metrics require raw-drawdown metrics")
    if not diagnostics_enabled and (
        ema_tail_enabled or raw_drawdown_enabled or raw_tail_enabled
    ):
        raise ValueError("HSL diagnostic feature outputs require diagnostics")
    if not diagnostics_enabled:
        if "#ifndef PASSIVBOT_HSL_DIAGNOSTICS_ENABLED" not in source:
            raise RuntimeError("MPS source is missing the HSL diagnostics feature guard")
        source = _HSL_DIAGNOSTICS_DISABLE_DEFINE + source
    source = _with_hsl_ema_tail(source, ema_tail_enabled)
    if raw_drawdown_enabled:
        if "#ifndef PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED" not in source:
            raise RuntimeError("MPS source is missing the HSL raw-drawdown feature guard")
        source = _HSL_RAW_DRAWDOWN_DEFINE + source
    if raw_tail_enabled:
        if "#ifndef PASSIVBOT_HSL_RAW_TAIL_ENABLED" not in source:
            raise RuntimeError("MPS source is missing the HSL raw-tail feature guard")
        source = _HSL_RAW_TAIL_DEFINE + source
    return source


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
        raise RuntimeError(
            "MPS source is missing the shared entry-interval contract"
        )
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
    values = np.ascontiguousarray(
        np.asarray(btc_prices, dtype=np.float32).reshape(-1)
    )
    if len(values) != int(expected_count):
        raise ValueError(
            "MPS BTC-risk prices must match the prepared candle count: "
            f"btc={len(values)}, candles={expected_count}"
        )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "MPS BTC-risk prices must remain finite and positive after float32 packing"
        )
    return torch.as_tensor(values, dtype=torch.float32, device="mps")


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
            and all(predicate(matrix[:, offset + key_index]) for offset in active_offsets)
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
    """Append the exact-default WEL sentinel to legacy two-side rows."""

    if params.ndim != 2:
        return params
    legacy_width = side_width - 1
    if params.shape[1] != legacy_width * 2:
        return params
    sentinel = np.full((params.shape[0], 1), -1.0, dtype=params.dtype)
    return np.concatenate(
        (
            params[:, :legacy_width],
            sentinel,
            params[:, legacy_width:],
            sentinel,
        ),
        axis=1,
    )


def _scale_directional_minute_parameters(
    params: np.ndarray,
    keys: tuple[str, ...],
    *,
    sides: int,
    interval_minutes: float,
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
        raise ValueError(
            "MPS candle interval must be finite and at least one minute"
        )
    scaled = np.array(params, dtype=np.float64, copy=True)
    minute_keys = {
        "ema_span_0",
        "ema_span_1",
        "entry_cooldown_minutes",
        "hsl_cooldown_minutes_after_red",
    }
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
        for key in minute_keys:
            scaled[:, offset + keys.index(key)] /= interval_minutes
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
    for column in minute_columns | {hsl_start_column + 3}:
        finite = np.isfinite(scaled[:, column])
        scaled[finite, column] /= interval_minutes
    if interval_minutes != 1.0:
        hsl_span_column = hsl_start_column + 2
        finite = np.isfinite(scaled[:, hsl_span_column])
        hsl_spans = scaled[finite, hsl_span_column]
        if np.any(hsl_spans < 1.0):
            raise ValueError(
                "MPS HSL EMA span override must be at least one minute"
            )
        alpha_1m = 2.0 / (hsl_spans + 1.0)
        decay_1m = 1.0 - alpha_1m
        alpha_per_candle = np.ones_like(decay_1m)
        positive_decay = decay_1m > 0.0
        alpha_per_candle[positive_decay] = -np.expm1(
            interval_minutes * np.log(decay_1m[positive_decay])
        )
        scaled[finite, hsl_span_column] = 2.0 / alpha_per_candle - 1.0
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

    override_keys = tuple(
        key for key, _path in TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS
    )
    return _scale_multicoin_coin_overrides(
        coin_overrides,
        interval_minutes,
        expected_cols=TRAILING_MARTINGALE_COIN_OVERRIDE_COLS,
        label="Trailing Martingale",
        minute_columns={
            override_keys.index("ema_span_0"),
            override_keys.index("ema_span_1"),
            override_keys.index("volatility_ema_span_1m"),
            TRAILING_MARTINGALE_COIN_OVERRIDE_COOLDOWN_COLUMN,
        },
        hsl_start_column=TRAILING_MARTINGALE_COIN_OVERRIDE_HSL_START_COLUMN,
    )


def _scalar_column_or_zero(scalars, index: int):
    if scalars.shape[1] > index:
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
            "MPS entry-interval stats have an invalid shape: "
            f"{tuple(stats.shape)}"
        )
    if counts.ndim != 2 or counts.shape[1] != MPS_ENTRY_INTERVAL_COUNT_COLS:
        raise RuntimeError(
            "MPS entry-interval counts have an invalid shape: "
            f"{tuple(counts.shape)}"
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
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_source_py(),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=4)
def _ema_anchor_long_no_hsl_shader_library(
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    source = _with_recovery_distribution(
        passivbot_rust.mps_ema_anchor_long_no_hsl_source_py(),
        recovery_distribution_enabled,
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=4)
def _ema_anchor_short_no_hsl_shader_library(
    recovery_distribution_enabled: bool = False,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    source = _with_recovery_distribution(
        passivbot_rust.mps_ema_anchor_short_no_hsl_source_py(),
        recovery_distribution_enabled,
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return torch.mps.compile_shader(source)


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
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
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
        diagnostics_enabled=hsl_diagnostics_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return torch.mps.compile_shader(source)


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
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
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
        diagnostics_enabled=hsl_diagnostics_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return torch.mps.compile_shader(source)


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
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
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
        diagnostics_enabled=hsl_diagnostics_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return torch.mps.compile_shader(source)


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
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
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
    return torch.mps.compile_shader(source)


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
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
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
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=32)
def _ema_anchor_multicoin_shader_library(
    hsl_ema_tail_enabled: bool = False,
    hsl_raw_drawdown_enabled: bool = False,
    hsl_raw_tail_enabled: bool = False,
    recovery_distribution_enabled: bool = False,
    dynamic_wel_by_tradability: bool = True,
    btc_risk_enabled: bool = False,
    equity_balance_diff_enabled: bool = False,
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_ema_anchor_multicoin_source_py(),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_dynamic_wel_by_tradability(
        source, dynamic_wel_by_tradability
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    return torch.mps.compile_shader(source)


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
):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    source = _with_hsl_features(
        passivbot_rust.mps_trailing_martingale_multicoin_source_py(),
        ema_tail_enabled=hsl_ema_tail_enabled,
        raw_drawdown_enabled=hsl_raw_drawdown_enabled,
        raw_tail_enabled=hsl_raw_tail_enabled,
    )
    source = _with_recovery_distribution(source, recovery_distribution_enabled)
    source = _with_dynamic_wel_by_tradability(
        source, dynamic_wel_by_tradability
    )
    source = _with_btc_risk(source, btc_risk_enabled)
    source = _with_equity_balance_diff(source, equity_balance_diff_enabled)
    source = _with_entry_interval(source, entry_interval_enabled)
    return torch.mps.compile_shader(source)


@lru_cache(maxsize=1)
def _strategy_eq_recovery_distribution_shader_library():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this process")
    import passivbot_rust

    return torch.mps.compile_shader(
        passivbot_rust.mps_strategy_eq_recovery_distribution_source_py()
    )


@lru_cache(maxsize=2)
def _strategy_eq_recovery_distribution_buffers(
    batch_size: int, sample_capacity: int
):
    shape = (int(batch_size), int(sample_capacity))
    return (
        torch.empty(shape, dtype=torch.int32, device="mps"),
        torch.empty(shape, dtype=torch.int32, device="mps"),
        torch.empty(
            (int(batch_size), MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device="mps",
        ),
        torch.tensor(shape, dtype=torch.int32, device="mps"),
    )


def strategy_eq_recovery_distribution_from_samples(
    strategy_equity_samples, *, sample_interval_days: float = 1.0
):
    """Approximate exact recovery summaries from uniformly spaced proxy samples."""

    if strategy_equity_samples.device.type != "mps":
        raise ValueError("strategy-equity recovery distribution requires an MPS tensor")
    if strategy_equity_samples.dtype != torch.float32:
        raise ValueError("strategy-equity recovery distribution requires float32 input")
    if strategy_equity_samples.ndim != 2:
        raise ValueError(
            "strategy-equity recovery distribution expects a batch-by-day matrix"
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
            device="mps",
        )
    if sample_capacity == 0:
        return torch.zeros(
            (batch_size, MPS_STRATEGY_EQ_RECOVERY_METRIC_COLS),
            dtype=torch.float32,
            device="mps",
        )
    stack, histogram, output, sizes = _strategy_eq_recovery_distribution_buffers(
        batch_size, sample_capacity
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


def _decode_outputs(daily, scalars, gaps) -> dict:
    active_days = torch.isfinite(daily[:, :, 1]) & (daily[:, :, 1] < float("inf"))

    def timestamp_column(index: int):
        values = scalars[:, index]
        return torch.where(
            values >= 0.0, values, torch.full_like(values, float("nan"))
        )

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
        "held_count": scalars[:, 30],
        "account_recovery_max_ms": scalars[:, 31],
        "hsl_long_enabled": scalars[:, 32] > 0.0,
        "hsl_short_enabled": scalars[:, 33] > 0.0,
        "hsl_triggers_long": scalars[:, 34],
        "hsl_triggers_short": scalars[:, 35],
        "hsl_restarts_long": scalars[:, 36],
        "hsl_restarts_short": scalars[:, 37],
        "hsl_tier_samples_total": scalars[:, 38],
        "hsl_tier_samples_yellow": scalars[:, 39],
        "hsl_tier_samples_orange": scalars[:, 40],
        "hsl_tier_samples_red": scalars[:, 41],
        "hsl_duration_sum_steps": scalars[:, 42],
        "hsl_duration_max_steps": scalars[:, 43],
        "hsl_duration_count": scalars[:, 44],
        "hsl_trigger_drawdown_sum": scalars[:, 45],
        "hsl_trigger_drawdown_count": scalars[:, 46],
        "hsl_flatten_time_sum_steps": scalars[:, 47],
        "hsl_flatten_time_count": scalars[:, 48],
        "hsl_restart_retrigger_count": scalars[:, 49],
        "hsl_halt_to_restart_equity_loss": scalars[:, 50],
        "hsl_panic_close_loss_sum": scalars[:, 51],
        "hsl_panic_close_loss_max": scalars[:, 52],
        "hsl_panic_loss_drawdown_min": scalars[:, 53],
        "hsl_panic_loss_drawdown_sum": scalars[:, 54],
        "hsl_panic_loss_drawdown_max": scalars[:, 55],
        "hsl_panic_loss_drawdown_count": scalars[:, 56],
        "hsl_drawdown_ema_max_long": scalars[:, 57],
        "hsl_drawdown_ema_max_short": scalars[:, 58],
        "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 59],
        "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 60],
        "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 61
        ),
        "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 62
        ),
        "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 63),
        "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 64),
        "hsl_drawdown_raw_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 65
        ),
        "hsl_drawdown_raw_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 66
        ),
    }
    output.update(_decode_btc_risk_outputs(daily, active_days, 9))
    return output


def _decode_multicoin_fused_outputs(daily, scalars, gaps) -> dict:
    output = _decode_outputs(daily, scalars, gaps)
    long_entry_initial_balance_pct = output.pop("entry_initial_balance_pct")
    output.update(
        {
            "entry_initial_balance_pct_long": long_entry_initial_balance_pct,
            "entry_initial_balance_pct_short": scalars[:, 59],
            "profit_sum_long": scalars[:, 60],
            "loss_sum_long": scalars[:, 61],
            "profit_sum_short": scalars[:, 62],
            "loss_sum_short": scalars[:, 63],
            "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 64],
            "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 65],
            "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
                scalars, 66
            ),
            "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
                scalars, 67
            ),
            "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 68),
            "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 69),
            "hsl_drawdown_raw_mean_worst_1pct_long": _scalar_column_or_zero(
                scalars, 70
            ),
            "hsl_drawdown_raw_mean_worst_1pct_short": _scalar_column_or_zero(
                scalars, 71
            ),
        }
    )
    return output


def _decode_directional_outputs(daily, scalars, gaps) -> dict:
    active_days = torch.isfinite(daily[:, :, 1]) & (daily[:, :, 1] < float("inf"))

    def timestamp_column(index: int):
        values = scalars[:, index]
        return torch.where(
            values >= 0.0, values, torch.full_like(values, float("nan"))
        )

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
        "hsl_tier_samples_yellow": scalars[:, 25],
        "hsl_tier_samples_orange": scalars[:, 26],
        "hsl_tier_samples_red": scalars[:, 27],
        "hsl_duration_sum_steps": scalars[:, 28],
        "hsl_duration_max_steps": scalars[:, 29],
        "hsl_duration_count": scalars[:, 30],
        "hsl_trigger_drawdown_sum": scalars[:, 31],
        "hsl_trigger_drawdown_count": scalars[:, 32],
        "hsl_flatten_time_sum_steps": scalars[:, 33],
        "hsl_flatten_time_count": scalars[:, 34],
        "hsl_restart_retrigger_count": scalars[:, 35],
        "hsl_halt_to_restart_equity_loss": scalars[:, 36],
        "hsl_panic_close_loss_sum": scalars[:, 37],
        "hsl_panic_close_loss_max": scalars[:, 38],
        "hsl_panic_loss_drawdown_min": scalars[:, 39],
        "hsl_panic_loss_drawdown_sum": scalars[:, 40],
        "hsl_panic_loss_drawdown_max": scalars[:, 41],
        "hsl_panic_loss_drawdown_count": scalars[:, 42],
        "profit_sum": scalars[:, 43],
        "loss_sum": scalars[:, 44],
        "position_unchanged_max_ms": scalars[:, 45],
        "entry_initial_balance_pct_long": scalars[:, 46],
        "entry_initial_balance_pct_short": scalars[:, 47],
        "total_wallet_exposure_max": scalars[:, 48],
        "total_wallet_exposure_mean": scalars[:, 49],
        "fill_count": scalars[:, 50],
        "fill_count_entry": scalars[:, 51],
        "fill_count_long": scalars[:, 52],
        "fills_active_days_count": scalars[:, 53],
        "pnl_recovery_max_ms": scalars[:, 54],
        "held_sum_ms": scalars[:, 55],
        "held_count": scalars[:, 56],
        "account_recovery_max_ms": scalars[:, 57],
        "profit_sum_long": scalars[:, 58],
        "loss_sum_long": scalars[:, 59],
        "profit_sum_short": scalars[:, 60],
        "loss_sum_short": scalars[:, 61],
        "hsl_drawdown_ema_max_long": scalars[:, 62],
        "hsl_drawdown_ema_max_short": scalars[:, 63],
        "hsl_strategy_eq_recovery_max_ms_long": scalars[:, 64],
        "hsl_strategy_eq_recovery_max_ms_short": scalars[:, 65],
        "hsl_drawdown_ema_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 66
        ),
        "hsl_drawdown_ema_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 67
        ),
        "hsl_drawdown_raw_max_long": _scalar_column_or_zero(scalars, 68),
        "hsl_drawdown_raw_max_short": _scalar_column_or_zero(scalars, 69),
        "hsl_drawdown_raw_mean_worst_1pct_long": _scalar_column_or_zero(
            scalars, 70
        ),
        "hsl_drawdown_raw_mean_worst_1pct_short": _scalar_column_or_zero(
            scalars, 71
        ),
    }
    output.update(_decode_btc_risk_outputs(daily, active_days, 8))
    return output


class MpsEmaAnchorRunner:
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
            raise ValueError(
                "max_realized_loss_pct must be finite and non-negative"
            )
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        self.loss_gate_enabled = encoded_max_realized_loss_pct < 1.0
        self.market_orders_allowed = bool(market_orders_allowed)
        taker_fee = market.maker_fee if taker_fee is None else float(taker_fee)
        market_order_slippage_pct = float(market_order_slippage_pct)
        market_order_near_touch_threshold = float(
            market_order_near_touch_threshold
        )
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
        self.rolling_capacity = (
            MPS_DIRECTIONAL_HSL_ROLLING_CAPACITY
            if pnl_lookback_bars > 0
            else 1
        )
        self.n = int(data["n"])
        self.n_days = int(data["n_days"])
        self.btc_prices = _btc_risk_price_tensor(
            btc_prices, expected_count=self.n
        )
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
        self.recovery_stride = (
            max(1, int(np.ceil(3_600_000.0 / float(run.interval_ms))))
            if self.recovery_distribution_enabled
            else 0
        )
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
            .to(dtype=torch.float32, device="mps")
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
            .to(device="mps")
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
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._rolling_buffers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
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
        return np.ascontiguousarray(scaled, dtype=np.float32)

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
                        device="mps",
                    ),
                    torch.zeros(
                        (
                            batch_size,
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
                            ),
                        ),
                        dtype=torch.float32,
                        device="mps",
                    ),
                    torch.zeros(
                        (batch_size, GAP_BINS), dtype=torch.int32, device="mps"
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

    def _hsl_rolling_buffers(self, batch_size: int):
        if batch_size not in self._rolling_buffers:
            # The kernel overwrites every active ring/deque slot.  Avoid
            # zeroing this large scratch allocation between generations.
            shape = (batch_size, 2, self.rolling_capacity, 2)
            self._rolling_buffers = {
                batch_size: (
                    torch.empty(shape, dtype=torch.float32, device="mps"),
                    torch.empty(shape, dtype=torch.int32, device="mps"),
                )
            }
        return self._rolling_buffers[batch_size]

    def _recovery_sample_buffer(self, batch_size: int):
        if batch_size not in self._recovery_buffers:
            self._recovery_buffers = {
                batch_size: torch.full(
                    (batch_size, self.n_recovery_samples),
                    float("nan"),
                    dtype=torch.float32,
                    device="mps",
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
                    device="mps",
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
            self.rolling_capacity,
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
        started = time.perf_counter() if profile else 0.0
        matrix = self._pack_params(params)
        packed = time.perf_counter() if profile else 0.0
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        rolling_pnl_values, rolling_pnl_indices = self._hsl_rolling_buffers(
            batch_size
        )
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
                device="mps",
            )
        prepared = time.perf_counter() if profile else 0.0
        loader, library_args = self._shader_library_cache_call()
        library, cold = _cached_library_with_miss(loader, *library_args)
        compiled = time.perf_counter() if profile else 0.0
        if profile:
            torch.mps.synchronize()
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
                rolling_pnl_values,
                rolling_pnl_indices,
            )
            if self.recovery_distribution_enabled:
                kernel_args += (recovery_samples,)
            library.passivbot_ema_anchor(
                *kernel_args,
                threads=(batch_size, 1, 1),
            )

        dispatch_once()
        if profile:
            torch.mps.synchronize()
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
        output = _decode_directional_outputs(daily, scalars, gaps)
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        if profile:
            torch.mps.synchronize()
            self.last_profile["metric_decode_seconds"] = (
                time.perf_counter() - finished
            )
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
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
    ):
        if side not in {"long", "short"}:
            raise ValueError(
                f"MPS multicoin runner side must be long or short, got {side!r}"
            )
        self.side = side
        self.collect_coin_fill_counts = bool(collect_coin_fill_counts)
        self.hsl_ema_tail_enabled = bool(hsl_ema_tail_enabled)
        self.hsl_raw_drawdown_enabled = bool(hsl_raw_drawdown_enabled)
        self.hsl_raw_tail_enabled = bool(hsl_raw_tail_enabled)
        self.recovery_distribution_enabled = bool(recovery_distribution_enabled)
        self.dynamic_wel_by_tradability = bool(dynamic_wel_by_tradability)
        fused = self.scalar_cols == MPS_MULTICOIN_FUSED_SCALAR_COLS
        if self.hsl_raw_tail_enabled:
            self.scalar_cols = (
                MPS_MULTICOIN_FUSED_SCALAR_COLS
                if fused
                else MPS_MULTICOIN_SCALAR_COLS
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
        self.btc_prices = _btc_risk_price_tensor(
            btc_prices, expected_count=self.n
        )
        self.equity_balance_diff_enabled = bool(equity_balance_diff_enabled)
        self.entry_interval_enabled = bool(entry_interval_enabled)
        if self.entry_interval_enabled and self.coin_override_label != "Trailing Martingale":
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
        self.daily_cols = MPS_MULTICOIN_DAILY_COLS + (
            3 if self.btc_risk_enabled else 0
        )
        self.recovery_stride = (
            max(1, int(np.ceil(3_600_000.0 / float(run.interval_ms))))
            if self.recovery_distribution_enabled
            else 0
        )
        self.n_recovery_samples = (
            max(
                1,
                (self.n + self.recovery_stride - 1) // self.recovery_stride + 1,
            )
            if self.recovery_distribution_enabled
            else 1
        )
        self.bars = data["bars"]
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
        self.coin_overrides = torch.as_tensor(
            self._prepare_coin_overrides(coin_overrides), device="mps"
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
            raise ValueError(
                "max_realized_loss_pct must be finite and non-negative"
            )
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
        market_order_near_touch_threshold = float(
            market_order_near_touch_threshold
        )
        if (
            not np.isfinite(market_order_near_touch_threshold)
            or market_order_near_touch_threshold < 0.0
        ):
            raise ValueError(
                "market_order_near_touch_threshold must be finite and non-negative"
            )
        liq_floor = max(0.0, run.starting_balance) * max(
            0.0, run.liquidation_threshold
        )
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
            ],
            dtype=torch.float32,
            device="mps",
        )
        self._buffers: dict[int, tuple[torch.Tensor, ...]] = {}
        self._recovery_buffers: dict[int, torch.Tensor] = {}
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
                        device="mps",
                    ),
                    torch.zeros(
                        (batch_size, self.scalar_cols),
                        dtype=torch.float32,
                        device="mps",
                    ),
                    torch.zeros(
                        (batch_size, GAP_BINS), dtype=torch.int32, device="mps"
                    ),
                    torch.zeros(
                        (batch_size, self.n_coins)
                        if self.collect_coin_fill_counts
                        else (1,),
                        dtype=torch.float32,
                        device="mps",
                    ),
                )
            }
        else:
            for buffer in self._buffers[batch_size]:
                buffer.zero_()
        self._buffers[batch_size][0][:, :, 1].fill_(float("inf"))
        self._buffers[batch_size][0][:, :, 5].fill_(float("inf"))
        return self._buffers[batch_size]

    def _end_steps(self, end_steps: np.ndarray | None, batch_size: int):
        if end_steps is None:
            if batch_size not in self._full_end_steps:
                self._full_end_steps[batch_size] = torch.full(
                    (batch_size,), self.n - 1, dtype=torch.int32, device="mps"
                )
            return self._full_end_steps[batch_size]
        values = np.asarray(end_steps, dtype=np.int32)
        if values.shape != (batch_size,):
            raise ValueError(
                f"expected one multi-coin end step per candidate, got {values.shape}"
            )
        values = np.clip(values, 1, self.n - 1)
        return torch.as_tensor(
            np.ascontiguousarray(values), dtype=torch.int32, device="mps"
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
        library.passivbot_ema_anchor_multicoin(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )

    def _library(self):
        loader, args = self._library_cache_call()
        return loader(*args)

    def _library_cache_call(self):
        return _ema_anchor_multicoin_shader_library, (
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.hsl_raw_tail_enabled,
            self.recovery_distribution_enabled,
            self.dynamic_wel_by_tradability,
            self.btc_risk_enabled,
            self.equity_balance_diff_enabled,
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_outputs(daily, scalars, gaps)

    def _recovery_sample_buffer(self, batch_size: int):
        if batch_size not in self._recovery_buffers:
            self._recovery_buffers[batch_size] = torch.full(
                (batch_size, self.n_recovery_samples),
                float("nan"),
                dtype=torch.float32,
                device="mps",
            )
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
                    device="mps",
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
                    device="mps",
                )
            }
            self._entry_interval_count_buffers = {
                batch_size: torch.zeros(
                    (batch_size, MPS_ENTRY_INTERVAL_COUNT_COLS),
                    dtype=torch.int32,
                    device="mps",
                )
            }
        else:
            self._entry_interval_stat_buffers[batch_size].zero_()
            self._entry_interval_count_buffers[batch_size].zero_()
        return (
            self._entry_interval_stat_buffers[batch_size],
            self._entry_interval_count_buffers[batch_size],
        )

    def run(
        self,
        params: np.ndarray,
        *,
        profile: bool = False,
        end_steps: np.ndarray | None = None,
    ) -> dict:
        started = time.perf_counter() if profile else 0.0
        matrix = self._pack_params(params)
        packed = time.perf_counter() if profile else 0.0
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        end_steps_mps = self._end_steps(end_steps, batch_size)
        daily, scalars, gaps, coin_fill_counts = self._output_buffers(batch_size)
        recovery_samples = (
            self._recovery_sample_buffer(batch_size)
            if self.recovery_distribution_enabled
            else None
        )
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
                size_values.extend(
                    [self.recovery_stride, self.n_recovery_samples]
                )
            self._sizes[sizes_key] = torch.tensor(
                size_values,
                dtype=torch.int32,
                device="mps",
            )
        prepared = time.perf_counter() if profile else 0.0
        loader, library_args = self._library_cache_call()
        library, cold = _cached_library_with_miss(loader, *library_args)
        compiled = time.perf_counter() if profile else 0.0
        if profile:
            torch.mps.synchronize()
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
            batch_size=batch_size,
        )
        if profile:
            torch.mps.synchronize()
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
            }
        else:
            self.last_profile = {}
        output = self._decode(daily, scalars, gaps)
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        output.update(
            _decode_entry_interval_outputs(
                entry_interval_stats, entry_interval_counts
            )
        )
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        if self.collect_coin_fill_counts:
            output["coin_fill_counts"] = coin_fill_counts
        if profile:
            torch.mps.synchronize()
            self.last_profile["metric_decode_seconds"] = (
                time.perf_counter() - finished
            )
        return output


class MpsEmaAnchorMulticoinFusedRunner(MpsEmaAnchorMulticoinRunner):
    """Persistent dual-side shared-account EMA Anchor runner on Apple MPS."""

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
        hedge_mode: bool = True,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
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
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
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
        self.short_coin_overrides = torch.as_tensor(
            self._prepare_coin_overrides(short_coin_overrides), device="mps"
        )
        max_realized_loss_pct = float(max_realized_loss_pct)
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            max_realized_loss_pct
        )
        market_order_slippage_pct = float(market_order_slippage_pct)
        market_order_near_touch_threshold = float(
            market_order_near_touch_threshold
        )
        liq_floor = max(0.0, run.starting_balance) * max(
            0.0, run.liquidation_threshold
        )
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
            ],
            dtype=torch.float32,
            device="mps",
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
        library.passivbot_ema_anchor_multicoin_fused(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps)


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

    coin_override_cols = TRAILING_MARTINGALE_COIN_OVERRIDE_COLS
    coin_override_label = "Trailing Martingale"
    def _prepare_coin_overrides(self, coin_overrides: np.ndarray) -> np.ndarray:
        return _scale_tm_multicoin_coin_overrides(
            coin_overrides, self.interval_minutes
        )

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
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
    ):
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
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
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
        )
        return _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS, sides=1
        )

    def _library(self):
        loader, args = self._library_cache_call()
        return loader(*args)

    def _library_cache_call(self):
        return _trailing_martingale_multicoin_shader_library, (
            self.hsl_ema_tail_enabled,
            self.hsl_raw_drawdown_enabled,
            self.hsl_raw_tail_enabled,
            self.recovery_distribution_enabled,
            self.dynamic_wel_by_tradability,
            self.btc_risk_enabled,
            self.equity_balance_diff_enabled,
            self.entry_interval_enabled,
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
        library.passivbot_trailing_martingale_multicoin(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_outputs(daily, scalars, gaps)


class MpsTrailingMartingaleMulticoinFusedRunner(
    MpsTrailingMartingaleMulticoinRunner
):
    """Persistent dual-side shared-account Trailing Martingale runner on MPS."""

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
        hedge_mode: bool = True,
        dynamic_wel_by_tradability: bool = True,
        btc_prices: np.ndarray | None = None,
        btc_risk_enabled: bool | None = None,
        equity_balance_diff_enabled: bool = False,
        entry_interval_enabled: bool = False,
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
            dynamic_wel_by_tradability=dynamic_wel_by_tradability,
            btc_prices=btc_prices,
            btc_risk_enabled=btc_risk_enabled,
            equity_balance_diff_enabled=equity_balance_diff_enabled,
            entry_interval_enabled=entry_interval_enabled,
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
        self.short_coin_overrides = torch.as_tensor(
            self._prepare_coin_overrides(short_coin_overrides), device="mps"
        )
        encoded_max_realized_loss_pct = _encode_max_realized_loss_pct(
            float(max_realized_loss_pct)
        )
        liq_floor = max(0.0, run.starting_balance) * max(
            0.0, run.liquidation_threshold
        )
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
            ],
            dtype=torch.float32,
            device="mps",
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
        library.passivbot_trailing_martingale_multicoin_fused(
            *kernel_args,
            threads=(batch_size, 1, 1),
        )

    def _decode(self, daily, scalars, gaps) -> dict:
        return _decode_multicoin_fused_outputs(daily, scalars, gaps)


class MpsTrailingMartingaleRunner(MpsEmaAnchorRunner):
    """Persistent single-coin trailing-martingale runner on Apple MPS."""

    def __init__(
        self,
        *args,
        hsl_enabled: bool = True,
        hsl_diagnostics_enabled: bool = True,
        entry_interval_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
            current_hour_boundary_ms = (
                current_ts_ms // 3_600_000
            ) * 3_600_000
            next_window_start = max(
                0,
                (current_hour_boundary_ms - first_ts_ms) // interval_ms,
            )
            hour_boundary_bits[step] = (
                2
                | (4 if window_ready else 0)
                | (8 if next_window_start < step else 0)
            )
            last_hour_boundary_ms = current_hour_boundary_ms
        boundary_bits = torch.as_tensor(
            hour_boundary_bits, dtype=torch.int32, device="mps"
        )
        self.flags[:, 3].bitwise_or_(boundary_bits)

    def _shader_library(self):
        loader, args = self._shader_library_cache_call()
        return loader(*args)

    def _shader_library_cache_call(
        self,
        dispatch_features: tuple[
            bool, bool, bool, bool, bool, bool, bool
        ] | None = None,
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
        )

    def _entry_interval_buffers(self, batch_size: int):
        if not self.entry_interval_enabled:
            return None, None
        if batch_size not in self._entry_interval_stat_buffers:
            self._entry_interval_stat_buffers = {
                batch_size: torch.zeros(
                    (batch_size, MPS_ENTRY_INTERVAL_STAT_COLS),
                    dtype=torch.float32,
                    device="mps",
                )
            }
            self._entry_interval_count_buffers = {
                batch_size: torch.zeros(
                    (batch_size, MPS_ENTRY_INTERVAL_COUNT_COLS),
                    dtype=torch.int32,
                    device="mps",
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
        return _pack_tm_parameter_matrix(
            scaled, TRAILING_MARTINGALE_SINGLE_COIN_PARAM_KEYS, sides=2
        )

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
            first_hour_boundary_ms = (
                first_hour_ts_ms // 3_600_000
            ) * 3_600_000
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
                self.n_recovery_samples
                if self.recovery_distribution_enabled
                else 0
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
        params_mps = torch.as_tensor(matrix, device="mps")
        batch_size = int(matrix.shape[0])
        daily, scalars, gaps = self._output_buffers(batch_size)
        rolling_pnl_values, rolling_pnl_indices = self._hsl_rolling_buffers(
            batch_size
        )
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
                device="mps",
            )
        prepared = time.perf_counter() if profile else 0.0
        loader, library_args = self._shader_library_cache_call(dispatch_features)
        library, cold = _cached_library_with_miss(loader, *library_args)
        compiled = time.perf_counter() if profile else 0.0
        if profile:
            torch.mps.synchronize()
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
                rolling_pnl_values,
                rolling_pnl_indices,
            )
            if self.recovery_distribution_enabled:
                kernel_args += (recovery_samples,)
            library.passivbot_trailing_martingale(
                *kernel_args,
                threads=(batch_size, 1, 1),
            )

        dispatch_once()
        if profile:
            torch.mps.synchronize()
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
        output = _decode_directional_outputs(daily, scalars, gaps)
        output.update(_decode_equity_balance_diff_outputs(equity_balance_diff))
        output.update(
            _decode_entry_interval_outputs(
                entry_interval_stats, entry_interval_counts
            )
        )
        if self.recovery_distribution_enabled:
            output["strategy_eq_recovery_samples"] = recovery_samples[
                :, :effective_recovery_sample_count
            ]
            output["strategy_eq_recovery_sample_interval_days"] = (
                self.recovery_stride * self.run_config.interval_ms / 86_400_000.0
            )
        if profile:
            torch.mps.synchronize()
            self.last_profile["metric_decode_seconds"] = (
                time.perf_counter() - finished
            )
        return output
